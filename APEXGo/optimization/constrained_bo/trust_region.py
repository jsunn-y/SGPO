import math
import torch
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
# from .approximate_gp import *
from constrained_bo.gp_utils.approximate_gp import *
from constrained_bo.gp_utils.constrained_max_posterior_sampling import MaxPosteriorSampling
from typing import Any
# based on TuRBO State from BoTorch


@dataclass
class TrustRegionState:
    dim: int
    batch_size: int 
    center_point: torch.Tensor # center point in search space where TR is defined
    best_x: Any = "" # input space item associated with center point if applicable (ie smiles, trajectory, etc.)
    best_value: float = -float("inf") 
    length: float = 0.8 
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32 
    success_counter: int = 0
    success_tolerance: int = 10 
    restart_triggered: bool = False
    best_constraint_values: torch.Tensor = torch.ones(2,)*torch.inf
    idx: int = None

def update_state_unconstrained(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += len(Y_next) # 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True

def update_tr_length(state):
    # Update the length of the trust region according to 
    # success and failure counters 
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min: # Restart when trust region becomes too small 
        state.restart_triggered = True
    
    return state

def update_state_constrained(state, Y_next, C_next): ## TODO: check this
    ''' Method used to update the TuRBO state after each
        step of optimization. 
        
        Success and failure counters are updated accoding to 
        the objective values (Y_next) and constraint values (C_next) 
        of the batch of candidate points evaluated on the optimization step. 
        
        As in the original TuRBO paper, a success is counted whenver 
        any one of the new candidate points imporves upon the incumbent 
        best point. The key difference for SCBO is that we only compare points 
        by their objective values when both points are valid (meet all constraints). 
        If exactly one of the two points beinc compared voliates a constraint, the 
        other valid point is automatically considered to be better. If both points 
        violate some constraints, we compare them inated by their constraint values. 
        The better point in this case is the one with minimum total constraint violation 
        (the minimum sum over constraint values)'''

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next <= 0 
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor] 
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0: # if none of the candidates are valid
        # pick the point with minimum violation
        sum_violation = C_next.sum(dim=-1)
        min_violation = sum_violation.min()
        # if the minimum voilation candidate is smaller than the violation of the incumbent
        if min_violation < state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_violation.argmin()].item()
            state.best_constraint_values = C_next[sum_violation.argmin()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else: # if at least one valid candidate was suggested, 
          # throw out all invalid candidates 
          # (a valid candidate is always better than an invalid one)

        # Case 1: if best valid candidate found has a higher obj value that incumbent best
            # count a success, the obj valuse has been improved
        imporved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value)
        # Case 2: if incumbent best violates constraints
            # count a success, we now have suggested a point which is valid and therfore better
        obtained_validity = torch.all(state.best_constraint_values > 0)
        if imporved_obj or obtained_validity: # If Case 1 or Case 2
            # count a success and update the best value and constraint values 
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a fialure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counts
    state = update_tr_length(state) 

    return state   

def update_state(state, Y_next, C_next): 
    if C_next is None:
        return update_state_unconstrained(state, Y_next)
    else:
        return update_state_constrained(state, Y_next, C_next)


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points 
    Y,  # Evaluated scores
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling 
    num_restarts=10,
    raw_samples=256,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device('cuda'),
    absolute_bounds=None, 
    constraint_model_list=None,
):
    assert acqf in ("ts", "ei")
    if constraint_model_list is not None:
        assert acqf == "ts" # SCBO only works with ts
        constrained=True
    else:
        constrained=False
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = state.center_point
    lb, ub = absolute_bounds 

    weights = torch.ones_like(x_center)
    if (lb is not None) and (ub is not None):
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, lb, ub) 
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, lb, ub) 
    else:
        weights = weights * 8 
        tr_lb = x_center - weights * state.length / 2.0
        tr_ub = x_center + weights * state.length / 2.0 

    if acqf == "ei":
        try:
            ei = qExpectedImprovement(model.cuda(), Y.max().cuda() ) 
            X_next, _ = optimize_acqf(ei,bounds=torch.stack([tr_lb, tr_ub]).cuda(),q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)
        except: 
            acqf = 'ts'

    if acqf == "ts":
        dim = X.shape[-1]
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        sobol = SobolEngine(dim, scramble=True) 
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda() 
        # Create a perturbation mask 
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cuda()

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cuda()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points 
        # SCBO --> Sample on the candidate points using Constrained Max Posterior Sampling
        thompson_sampling = MaxPosteriorSampling(
            model=model,
            constraint_models=constraint_model_list,
            replacement=False,
            constrained=constrained,
        ) 
        with torch.no_grad():
            X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size )

    return X_next
