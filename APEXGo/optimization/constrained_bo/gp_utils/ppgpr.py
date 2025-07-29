import torch
torch.set_float32_matmul_precision("highest")
# ppgpr
from .base import DenseNetwork
import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior
# from lolbo.utils.bo_utils.censored_likelihood import CensoredGaussianLikelihood
from gpytorch.distributions import MultivariateNormal

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, likelihood):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1
        self.likelihood = likelihood 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode 
            self.likelihood.eval()

            dist = self.likelihood(self(X), censoring=torch.zeros(X.shape[0]), censored=False)  # Normal(loc: torch.Size([10, 5000]), scale: torch.Size([10, 5000]))

            return GPyTorchPosterior(dist)


# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood, hidden_dims=(32, 32), feature_extractor=None):
        if feature_extractor is None:
            feature_extractor = DenseNetwork(
                input_dim=inducing_points.size(-1),
                hidden_dims=hidden_dims,
            ).to(inducing_points.device)
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(GPModelDKL, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
        self, X, output_indices=None, observation_noise=False, *args, **kwargs
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode 
        self.likelihood.eval()

        dist = self.likelihood(self(X), censoring=torch.zeros(X.shape[0], device=torch.device("cuda")), censored=False)  # Normal(loc: torch.Size([10, 5000]), scale: torch.Size([10, 5000]))

        return GPyTorchPosterior(dist)
    
def mvn_sample(mvn: MultivariateNormal):
    function_samples = mvn.rsample().unsqueeze(0)

    # Standard sampling 
    if function_samples.isfinite().all():
        return function_samples

    # First failure, add jitter
    mvn_with_jitter = MultivariateNormal(mvn.loc, gpytorch.add_jitter(mvn.lazy_covariance_matrix, jitter_val=1e-5))
    function_samples = mvn_with_jitter.rsample().unsqueeze(0)
    
    if function_samples.isfinite().all():
        return function_samples

    # Second failure, convert to double precision with jitter
    mvn_double_precision = MultivariateNormal(mvn.loc.double(), gpytorch.add_jitter(mvn.lazy_covariance_matrix.double(), jitter_val=1e-5))
    function_samples = mvn_double_precision.rsample().unsqueeze(0)
    
    if function_samples.isfinite().all():
        return function_samples
    
    # This is wild just return the mean
    function_samples = mvn.loc.unsqueeze(0)
    
    return function_samples