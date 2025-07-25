import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor
from tqdm.auto import tqdm

# import models.MDLM.models as models
# import models.MDLM.noise_schedule as noise_schedule
# import models.MDLM.utils as utils
from models.pretraining.model.mdlm_diffusion import MDLMDiffusion, _sample_categorical
from .base import GenerativeModel
LOG2 = math.log(2)


class MDLM(GenerativeModel):
    def __init__(self, model_name, config, tokenizer, num_steps=128, seq_len=389, device='cuda'):
        best_checkpoint = os.path.join("checkpoints", model_name, "best_model.ckpt")
        self.tokenizer = hydra.utils.instantiate(tokenizer, sequences=True)
        # self.model = MDLMDiffusion(config=config, tokenizer=tokenizer)
        # self.model.load_state_dict(torch.load(best_checkpoint, weights_only=False)['state_dict'])
        # checkpoint = torch.load(best_checkpoint, weights_only=False)
        # state_dict_keys = checkpoint['state_dict'].keys()
        # print(state_dict_keys)

        self.model = MDLMDiffusion.load_from_checkpoint(best_checkpoint, tokenizer=tokenizer, config=config)
        #print(self.model.state_dict().keys()) 
        self.model.set_length(seq_len)
        #self.mask_index = self.tokenizer.mask_id #or can use self.model.mask_index
        self.S = self.model.vocab_size
        #self.S = self.tokenizer.K
        self.timestep = num_steps
        self.length = seq_len
        self.device = str(device).split(":")[-1]
        
        #TODO: make this part of the config
        self.diffusion = config.diffusion #"absorbing_state"
        self.parameterization = config.parameterization #"subs"
        
        # self.model.ema=None
    def sample(self, num_samples):
        self.model.config.eval_batch_size = num_samples
        # return self.model.restore_model_and_sample(num_steps=128)
        return self.model._sample(num_steps=self.timestep)
    
    def get_start(self, batch_size):
        return self.model.mask_index * torch.ones((batch_size, self.length), dtype=torch.int64).to(self.device)
    
    @torch.no_grad()
    def p_sample(self, x, t, t_next=None, hard=True, guidance=None):
        t = t/self.timestep
        t_next = t/self.timestep # parameterization is in [0,1]
        sigma_t, _ = self.model.noise(t)
        sigma_s, _ = self.model.noise(t_next)
        sigma_t = sigma_t * torch.ones(x.shape[0]).to(x.device)
        sigma_s = sigma_s * torch.ones(x.shape[0]).to(x.device)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.model.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim

        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        if self.diffusion == "absorbing_state":
            q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
            q_xs[:, :, self.model.mask_index] = move_chance_s[:, :, 0]
            if hard:
                _x = _sample_categorical(q_xs)
                copy_flag = (x != self.model.mask_index).to(x.dtype)
                return copy_flag * x + (1 - copy_flag) * _x
            else:
                return q_xs
        elif self.diffusion == "uniform":
            q_xs = self.model._compute_posterior(
                x=log_p_x0.exp(),
                xt=x,
                alpha_s=1 - move_chance_s,
                alpha_t=1 - move_chance_t)#.log()
            if hard:
                return _sample_categorical(q_xs)
            return q_xs
        else:
            raise NotImplementedError(
                f"Diffusion type {self.diffusion} not implemented.")

    def sigma(self, t):
        sigma, _ = self.model.noise(t/self.timestep)
        return sigma
    
    def q_sample(self, x, t):
        t = t/self.timestep
        sigma, dsigma = self.model.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None]).to(x.device)
        xt = self.model.q_xt(x, move_chance)
        return xt
        
    def get_all_jump_transitions(
        self,
        x: torch.Tensor,  # Shape: (B, D)
    ) -> torch.Tensor:  # Shape: (B*D*S, D)
        """
        Gets all possible single-dimension transitions from current states.

        Creates a tensor containing all possible states that differ from input states
        in exactly one position, for each sequence in the batch.

        Args:
            xt: Current state tensor of shape (batch_size, sequence_length)
            S: Size of categorical state space (number of possible values per position)

        Returns:
            Tensor of shape (batch_size * sequence_length * state_space, sequence_length)
            containing all possible single-token transitions
        """
        B, D = x.shape
        device = x.device

        # Create B*D*S copies of input sequences
        # Shape: (B, 1, D) -> (B, D*S, D)
        xt_expand = x.unsqueeze(1).repeat(1, D * self.S, 1)
        # Flatten batch and transition dimensions
        # Shape: (B, D*S, D) -> (B*D*S, D)
        xt_expand = xt_expand.view(-1, D)

        # Create indices for all possible transitions
        # Shape: (D*S,) -> (B, D*S) -> (B*D*S,)
        jump_idx = torch.arange(D * self.S).to(device)
        jump_idx = jump_idx.repeat(B, 1).flatten()

        # Create tensor for states after one transition
        xt_jumps = xt_expand.clone()

        # Calculate which dimension changes for each transition
        # Shape: (B*D*S,)
        jump_dims = jump_idx // self.S

        # Calculate new value for changed dimension
        # Shape: (B*D*S,)
        jump_states = jump_idx % self.S

        # Apply transitions by assigning new values at transition dimensions
        # Shape: (B*D*S, D)
        xt_jumps[
            torch.arange(jump_idx.size(0), device=device),
            jump_dims,  # Index the transitioned dimension
        ] = jump_states  # Assign the new state

        return xt_jumps
    
    ### Modified from https://github.com/kuleshov-group/discrete-diffusion-guidance/blob/edb0f8c28b7caeb4ea7a06a2fee8d74ab6da1661/diffusion.py#L1458 ###

    # def _sample_prior(self, *batch_dims):
    #     if self.diffusion == 'absorbing_state':
    #         return self.mask_index * torch.ones(
    #         *batch_dims, dtype=torch.int64, device=self.device)
    #     if self.diffusion == 'uniform':
    #         return torch.randint(
    #         0, self.vocab_size, batch_dims, dtype=torch.int64,
    #         device=self.device)
    #     elif self.diffusion == 'uniform_data_marginals':
    #         if self.limiting_distribution.squeeze().ndim == 2:
    #             batch_dims = (batch_dims[0],)
    #         return torch.distributions.Categorical(
    #             self.limiting_distribution.squeeze()).sample(
    #             sample_shape=torch.Size(batch_dims))
    #     raise NotImplementedError(
    #         f'Diffusion type {self.diffusion} not '
    #         'implemented.')

    def _nos_denoise(
      self,
      classifier_model: torch.nn.Module, #TODO:make this more specific
      num_nos_steps: int,
      nos_step_size: float,
      nos_stability_coef: float,
      xt: torch.Tensor,
      t: torch.Tensor,
    ) -> typing.Tuple[torch.tensor, torch.tensor, None]:
        
        # num_nos_steps = nos_config.num_nos_steps
        # nos_step_size = nos_config.nos_step_size
        # nos_stability_coef = nos_config.nos_stability_coef
        # conditioning_class = nos_config.conditioning_class

        t = t/self.timestep
        t_next = t/self.timestep # parameterization is in [0,1]
        sigma_t, _ = self.model.noise(t)
        sigma_s, _ = self.model.noise(t_next)
        sigma_t = sigma_t * torch.ones(xt.shape[0]).to(xt.device)
        sigma_s = sigma_s * torch.ones(xt.shape[0]).to(xt.device)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        time_conditioning = sigma_t #added this, could share memory instead

        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        #print(self.model.backbone.state_dict().keys()) 
        # Compute original diffusion_log_probs and hidden states
        copy_flag = (xt != self.model.mask_index).to(torch.bool)
        with torch.no_grad():
            time_conditioning = self.model._process_sigma(time_conditioning)
        with torch.amp.autocast(device_type=self.device, dtype=torch.float32):
            logits, hidden_states = self.model.backbone(
            xt, time_conditioning, cond=None,
            return_hidden_states=True)
            hidden_states = [h.detach() for h in hidden_states] #added this line to prevent gradient flow

            if self.parameterization == 'subs':
                log_x_theta = self.model._subs_parameterization(
                logits=logits, xt=xt)
            elif self.parameterization == 'd3pm':
            # returns log_probs
                if self.subs_masking:  # Can use "zero masking prob"
                    logits[:, :,
                self.model.mask_index] += self.model.neg_infinity
                log_x_theta = logits.log_softmax(dim=-1)
            else:
                raise NotImplementedError(
                f"Parameterization {self.parameterization} not implemented for NOS guidance.")
            if self.diffusion == 'absorbing_state':
                diffusion_log_probs = log_x_theta + torch.log(
                1. - (move_chance_s / move_chance_t))
                diffusion_log_probs[..., self.model.mask_index] = torch.log(
                    move_chance_s / move_chance_t)[:, :, 0]
                diffusion_log_probs[copy_flag] = self.model.neg_infinity
                diffusion_log_probs[copy_flag, xt[copy_flag]] = 0.0
            elif self.diffusion == 'uniform':
                diffusion_log_probs = self.model._compute_posterior(
                    x=log_x_theta.exp(),
                    xt=xt,
                    alpha_s=1 - move_chance_s,
                    alpha_t=1 - move_chance_t).log()
            diffusion_log_probs = diffusion_log_probs.detach() #added this to prevent gradient flow

        # Perform NOS steps
        kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean') # #removed this to follow original NOS
        delta = torch.nn.Parameter(
        torch.zeros_like(hidden_states[-1]),
        requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=nos_step_size)
        with torch.enable_grad():
            for _ in range(num_nos_steps): #tqdm(, desc='NOS', leave=False):
                h_current = hidden_states[-1] + delta
                # target_loss = classifier_model.get_log_probs(
                # xt, time_conditioning, x_emb=h_current)[..., conditioning_class].sum()
                with torch.amp.autocast(device_type=self.device, dtype=torch.float32):
                    # target_loss = torch.nn.functional.log_softmax(classifier_model.forward(xt, time_conditioning, x_emb=h_current), dim=-1)[..., conditioning_class].sum() #check that the classifier outputs the same thing
                    # target_loss = torch.nn.functional.log_softmax(classifier_model.forward(xt, time_conditioning, x_emb=h_current)).sum() #for now we're doing regression, not sure if this works, check
                    target_loss = classifier_model.forward(xt, t*torch.ones(xt.shape[0], dtype=torch.long).to(xt.device)*self.timestep, x_emb=h_current).sum()  # This is log-probability - 200
                    #apply a log sigmoid
                    #target_loss = torch.nn.functional.logsigmoid(target_loss).sum()
                    new_logits = self.model.forward(xt, time_conditioning,
                                            cond=None, #used to be commented out
                                            x_emb=h_current)
                if self.diffusion == 'absorbing_state':
                    adjusted_log_probs = new_logits + torch.log(
                        1. - (move_chance_s / move_chance_t))
                    adjusted_log_probs[
                        ..., self.model.mask_index] = torch.log(
                        move_chance_s / move_chance_t)[:, :, 0]
                    adjusted_log_probs[
                        copy_flag] = self.model.neg_infinity
                    adjusted_log_probs[copy_flag, xt[copy_flag]] = 0.0
                    #adjusted_log_probs = adjusted_log_probs / 2 #added this line for smoothing
                elif self.diffusion == 'uniform':
                    adjusted_log_probs = self.model._compute_posterior(
                        x=new_logits.exp(),
                        xt=xt,
                        alpha_s=1 - move_chance_s,
                        alpha_t=1 - move_chance_t).log()
                else:
                    raise NotImplementedError(
                        f"Diffusion type {self.diffusion} not implemented.")
                #print(adjusted_log_probs)
                #print(diffusion_log_probs)
                # print("Entropy of adjusted_log_probs:", -(adjusted_log_probs.exp() * adjusted_log_probs).sum(dim=-1).mean())
                # print("Entropy of diffusion_log_probs:", -(diffusion_log_probs.exp() * diffusion_log_probs).sum(dim=-1).mean())

                kl = kl_loss(adjusted_log_probs, diffusion_log_probs)
                # entropy = -(adjusted_log_probs.exp() * adjusted_log_probs).sum(dim=-1).mean()
                
                #entropy_coef = 1
                # print("target_loss", target_loss)
                #print("kl", kl)
                #print(hidden_states[-1].mean())
                # print(delta.norm())
                loss = -target_loss + nos_stability_coef * kl # - entropy_coef * entropy
                #print(loss)
                optimizer.zero_grad()
                loss.backward() #retain_graph=True #added retain_graph
                optimizer.step()
                
        with torch.amp.autocast(device_type=self.device, dtype=torch.float32):
            guided_logits = self.model.forward(
                xt, time_conditioning,
                # cond=None,
                x_emb=hidden_states[-1] + delta.data)
        if self.diffusion == 'absorbing_state':
            diffusion_log_probs = guided_logits + torch.log(
            1. - (move_chance_s / move_chance_t))
            diffusion_log_probs[
                ..., self.model.mask_index] = torch.log(
                move_chance_s / move_chance_t)[:, :, 0]
            diffusion_log_probs.detach()
            guided_probs = diffusion_log_probs.exp()
        elif self.diffusion == 'uniform':
            guided_probs = self.model._compute_posterior(
                x=guided_logits.exp(),
                xt=xt,
                alpha_s=1 - move_chance_s,
                alpha_t=1 - move_chance_t).detach()
        else:
            raise NotImplementedError(
                f"Diffusion type {self.diffusion} not implemented.")

        xs = _sample_categorical(guided_probs)
        if self.diffusion == 'absorbing_state':
            xs = torch.where(copy_flag, xt, xs)

        return xs, guided_probs, None

    def get_guided_rates(
        self,
        predictor_model,
        x: torch.Tensor,  # Shape: (B, D)
        t: float,
        q_t: torch.Tensor,  # Shape: (B, D, S)
        use_tag: bool = False,
        guide_temp: float = 1.0,
        log_prob_ratio_cutoff: float = 80.0,
    ) -> torch.Tensor:
        """
        Get guided rates for classifier guidance.
        """
        if guide_temp == 0:
            return q_t
        else:
            B, D = x.shape
            device = x.device
            t = t * torch.ones((B,), device=device)
            if not use_tag:
                # Exact guidance case
                # log p(y|x=z_t), shape (B,)
                log_prob_xt = predictor_model(x, t)

                # Get all jump transitions, shape (B*D*S, D)
                xt_jumps = self.get_all_jump_transitions(x)

                # Get log probs for all transitions
                # Shape: (B*D*S,) -> (B, D, S)
                log_prob_xt_jumps = predictor_model(
                    xt_jumps, t.repeat(1, D * self.S).flatten()
                ).view(B, D, self.S)

                # Compute log ratios
                # Shape (B, D, S)
                log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

            else:
                # Taylor-approximated guidance (TAG) case
                # One-hot encode categorical data, shape (B, D, S)
                xt_ohe = F.one_hot(x.long(), num_classes=self.S).to(torch.float)

                # \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
                with torch.enable_grad():
                    xt_ohe.requires_grad_(True)
                    # log p(y|x=z_t), shape (B,)
                    log_prob_xt_ohe = predictor_model(xt_ohe, t)
                    log_prob_xt_ohe.sum().backward()
                    # Shape (B, D, S)
                    grad_log_prob_xt_ohe = xt_ohe.grad
                # 1st order Taylor approximation of the log difference
                # Shape (B, D, S)
                log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
                    dim=-1, keepdim=True
                )
            #check the rates here
            #print(log_prob_ratio[0,2, :]) #only mutated positions should be changing, everything else should be zero

            # Scale log prob ratio by temperature
            log_prob_ratio /= guide_temp

            # Clamp the log prob ratio to avoid overflow in exp
            log_prob_ratio = torch.clamp(log_prob_ratio, max=80)
            # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
            prob_ratio = torch.exp(log_prob_ratio)
            # Multiply the reverse rate elementwise with the density ratio
            # Note this doesn't deal with the diagonals
            # print(prob_ratio.max(), prob_ratio.min())
            # print(q_t[0,0], prob_ratio[0,0])
            q_t = q_t * prob_ratio
            if q_t.isnan().any():
                raise ValueError(f"The rate matrix 'q_t' contains NaNs.")

            return q_t
