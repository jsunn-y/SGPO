from abc import ABC, abstractmethod
from typing import Any, Optional
from unittest import runner

import torch
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.generation.utils import _flip_sub_unique
from botorch.models.model import Model
from torch import Tensor
from torch.nn import Module


# Code copied form botorch MaxPosteriorSampling class 
# https://botorch.org/api/_modules/botorch/generation/sampling.html


class SamplingStrategy(Module, ABC):
    r"""Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int = 1, **kwargs: Any) -> Tensor:
        r"""Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """

        pass  # pragma: no cover


# MaxPosteriorSampling class from botorch modified to support sampling with constraints
class MaxPosteriorSampling(SamplingStrategy):
    r"""Sample from a set of points according to their max posterior value.

    Example:
        >>> MPS = MaxPosteriorSampling(model)  # model w/ feature dim d=3
        >>> X = torch.rand(2, 100, 3)
        >>> sampled_X = MPS(X, num_samples=5) 
    """
    def __init__(
        self,
        model: Model,
        constraint_models: list = None,
        objective: Optional[MCAcquisitionObjective] = None,
        replacement: bool = True,
        constrained: bool = False,
    ) -> None:
        r"""Constructor for the SamplingStrategy base class.

        Args:
            model: A fitted model.
            objective: The objective. Typically, the AcquisitionObjective under which
                the samples are evaluated. If a ScalarizedObjective, samples from the
                scalarized posterior are used. Defaults to `IdentityMCObjective()`.
            replacement: If True, sample with replacement.
        """
        super().__init__()
        self.model = model
        if objective is None:
            objective = IdentityMCObjective()
        self.objective = objective
        self.replacement = replacement
        self.constraint_models = constraint_models
        self.constrained = constrained

    def forward(
        self, X: Tensor, num_samples: int = 1, observation_noise: bool = False, max_constr_val: int = 0,
    ) -> Tensor:
        r"""Sample from the model posterior.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension) according to the maximum posterior value under the objective.
            num_samples: The number of samples to draw.
            observation_noise: If True, sample with observation noise.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        # if isinstance(self.objective, ScalarizedObjective):
        #     posterior = self.objective(posterior)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))

        # SHAPES: (tested shapes in practice)
        #   X   =   N x d   =   torch.Size([5000, 256])
        #   samples  =  bsz x N x 1     =   orch.Size([10, 5000, 1])   constraits --> (10, 5000, c+1)

        # If we are using constraints 
        if self.constrained: 
            # Case 1: Multi-task Model where remaining tasks are constraints
            if (samples.shape[-1] > 1): 
                # NOTE: this assumes that model returns a vector where the first number is the objective 
                # function value and the remaining values are predicted constraint values 
                constraint_samples = samples[:,:,1:]  #  bsz x N x c   ie ([10, 5000, 1])
                samples = samples[:,:,0].unsqueeze(-1) #  bsz x N x 1   ie torch.Size([10, 5000, 1])
            # Case 2: Seperate Model for each Constraint (in list self.constraint_models) 
            elif self.constraint_models is not None:
                all_constraint_samples = []
                for constr_model in self.constraint_models:
                    constraint_posterior = constr_model.posterior(X, observation_noise=observation_noise)
                    constr_samples = constraint_posterior.rsample(sample_shape=torch.Size([num_samples]))
                    all_constraint_samples.append(constr_samples)
                constraint_samples = torch.cat(all_constraint_samples, dim=-1) 

            valid_samples = constraint_samples <= max_constr_val   # bsz x N x c  torch.Size([10, 5000, 1])
            if valid_samples.shape[-1] > 1: # more than one constraint
                valid_samples = torch.all(valid_samples, dim=-1).unsqueeze(-1) # # bsz x N x 1  (remains to be tested)
            # if all elements violate constraints
            if valid_samples.sum() == 0: 
                # if none of the samples meet the constraints 
                    # we pick the one that minimizes total violation... (By SCBO paper) 
                constraint_samples = constraint_samples.sum(dim=-1) # # bsz x N x c  --> bsz x N  (works for 1 constr or many)
                min_idxs = torch.argmin(constraint_samples, dim=-1)  # (bsz,)
                min_violators = X[min_idxs, :] # (bsz,d)  ie 10 x 256 
                return min_violators 
            # replace violators with -infinty so it will never choose them! 
            samples = torch.where(valid_samples, samples, -torch.inf*torch.ones(samples.shape).cuda()) # bsz x N x 1  ie torch.Size([8, 5000, 1])
        
        
        # if isinstance(self.objective, ScalarizedObjective):
        #     obj = samples.squeeze(-1)  # num_samples x batch_shape x N 
        # else:
        obj = self.objective(samples, X=X)  # num_samples x batch_shape x N
        if self.replacement:
            # if we allow replacement then things are simple(r)
            idcs = torch.argmax(obj, dim=-1)
        else:
            # if we need to deduplicate we have to do some tensor acrobatics
            # first we get the indices associated w/ the num_samples top samples
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            # generate some indices to smartly index into the lower triangle of
            # idcs_full (broadcasting across batch dimensions)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            # pick the unique indices in order - since we look at the lower triangle
            # of the index matrix and we don't sort, this achieves deduplication
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                # TODO: Find a better way to do this
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                # TODO: Find a general way to do this efficiently.
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )
        # idcs is num_samples x batch_shape, to index into X we need to permute for it
        # to have shape batch_shape x num_samples
        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)
        # in order to use gather, we need to repeat the index tensor d times
        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        # now if the model is batched batch_shape will not necessarily be the
        # batch_shape of X, so we expand X to the proper shape
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        # finally we can gather along the N dimension
        return torch.gather(Xe, -2, idcs) # means use the idcs to index dimension -2 