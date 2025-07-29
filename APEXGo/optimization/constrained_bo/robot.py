import torch
import gpytorch
import numpy as np
from gpytorch.mlls import PredictiveLogLikelihood 
from constrained_bo.trust_region import TrustRegionState, update_state, generate_batch
from constrained_bo.gp_utils.update_models import (
    update_surr_model, 
    update_constraint_surr_models,
    update_models_end_to_end_with_constraints,
)
from constrained_bo.gp_utils.ppgpr import GPModelDKL
import time

class RobotState:

    def __init__(
        self,
        M,
        tau,
        objective,
        train_x,
        train_y,
        train_c=None,
        train_z=None,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func='ts',
        verbose=True,
    ):

        self.tau                = tau               # Diversity threshold
        self.M                  = M                 # Number of diverse soltuions we seek
        self.objective          = objective         # Objective with vae and associated diversity function for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_c            = train_c           # initial train c data (for constraints, see lolrobot subclass)
        self.train_z            = train_z           # initial train z data (for latent space objectives, see lolrobot subclasss)
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose

        assert acq_func == "ts"
        if minimize:
            self.train_y = self.train_y * -1

        self.num_new_points = 0 # number of newly acquired points (in acquisiton)
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs

        self.initialize_global_surrogate_model()
        self.initialize_xs_to_scores_dict()
        self.initialize_tr_states()


    def search_space_data(self):
        return self.train_x

    def initialize_xs_to_scores_dict(self,):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict


    def initialize_tr_states(self):
        # if self.train_c is not None:  # if constrained 
        #     bool_arr = torch.all(self.train_c <= 0, dim=-1) # all constraint values <= 0
        #     vaid_train_y = self.train_y[bool_arr]
        #     valid_c_vals = self.train_c[bool_arr]
        # else:
        #     vaid_train_y = self.train_y
        #     best_constraint_values = None

        # if len(vaid_train_y) == 0:
        #     # best_value = -torch.inf 
        #     # if self.minimize:
        #     #     best_value = torch.inf
        #     if self.train_c is not None: 
        #         best_constraint_values = torch.ones(1,self.train_c.shape[1])*torch.inf
        # else:
        #     # best_value=torch.max(vaid_train_y).item()
        #     if self.train_c is not None: 
        #         best_constraint_values = valid_c_vals[torch.argmax(vaid_train_y)]
        #         if len(best_constraint_values.shape) == 1:
        #             best_constraint_values = best_constraint_values.unsqueeze(-1) 

        self.rank_ordered_trs = [] 
        for i in range(self.M):
            state = TrustRegionState( # initialize turbo state
                    dim=self.objective.dim,
                    batch_size=self.bsz, 
                    center_point=None,
                    # best_value=None, ## TODO: make sure this is right
                    # best_constraint_values=best_constraint_values, ## TODO: make sure this is right
                    # TODO: make sure each tr has its own constraint values is the right way to do this
                    idx=i, # track which tr this is, used for logging tr id for each candidate
                )
            self.rank_ordered_trs.append(state)
        
        # find feasible tr centers to start 
        self.recenter_trs() 


    def recenter_trs(self):
        # recenter trust regions and log best diverse set found 
        M_diverse_scores = []
        tr_center_xs = []
        idx_num = 0
        _, top_t_idxs = torch.topk(self.train_y.squeeze(), len(self.train_y))
        for ix, state in enumerate(self.rank_ordered_trs):
            while True: 
                # if we run out of feasible points in dataset
                if idx_num >= len(self.train_y): 
                    # Randomly sample a new feasible point (rare occurance)
                    center_x, center_point, center_score = self.randomly_sample_feasible_center(higher_ranked_xs=tr_center_xs) 
                    break
                # otherwise, finding highest scoring feassible point in remaining dataset for tr center
                center_idx = top_t_idxs[idx_num]
                center_score = self.train_y[center_idx].item()
                center_point = self.search_space_data()[center_idx] 
                center_x = self.train_x[center_idx]
                idx_num += 1
                if self.is_feasible(center_x, higher_ranked_xs=tr_center_xs):
                    break 

            tr_center_xs.append(center_x) 
            M_diverse_scores.append(center_score)
            state.center_point = center_point
            state.best_value = center_score
            state.best_x = center_x 

        self.M_diverse_scores = np.array(M_diverse_scores)
        self.M_diverse_xs = tr_center_xs


    def restart_trs_as_needed(self):
        for ix, state in enumerate(self.rank_ordered_trs):
            if state.restart_triggered:
                new_state = TrustRegionState( 
                    dim=self.objective.dim,
                    batch_size=self.bsz, 
                    center_point=state.center_point,
                    best_value=state.best_value,
                    best_x=state.best_x
                )
                self.rank_ordered_trs[ix] = new_state


    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for i in range(self.train_c.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
            n_pts = min(self.train_z.shape[0], 1024)
            c_model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
            c_mll = PredictiveLogLikelihood(c_model.likelihood, c_model, num_data=self.train_z.size(-2))
            c_model = c_model.eval() 
            # c_model = self.model.cuda()
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self


    def initialize_global_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        n_pts = min(self.search_space_data().shape[0], 1024)
        self.model = GPModelDKL(self.search_space_data()[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.search_space_data().size(-2))
        self.model = self.model.eval() 
        self.model = self.model.cuda()

        if self.train_c is not None:
            self.initialize_constraint_surrogates()
        else:
            self.c_models = None
            self.c_mlls = None

        return self


    def update_surrogate_model(self ): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            X = self.search_space_data() # this is just self.train_x
            Y = self.train_y.squeeze(-1)
            # considering constraints
            train_c = self.train_c
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            X = self.search_space_data()[-self.num_new_points:]
            Y = self.train_y[-self.num_new_points:].squeeze(-1)
            # considering constraints
            if self.train_c is not None:
                train_c = self.train_c[-self.num_new_points:]
            else:
                train_c = None 
            
        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            X,
            Y,
            n_epochs
        )

        # considering constraints
        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                self.c_models,
                self.c_mlls,
                self.learning_rte,
                X,
                train_c,
                n_epochs
            )

        self.initial_model_training_complete = True


    def is_feasible(self, x, higher_ranked_xs): 
        current_time = time.time()
        for higher_ranked_x in higher_ranked_xs:
            if self.objective.divf(x, higher_ranked_x) < self.tau:
                return False 
        # print(f"checking candidate for diversity in {time.time() - current_time}")
        # also make sure candidate satisfies constraints
        if isinstance(x, str):
            seq = [x] 
        else:
            seq = x
        # This is already a string, so no need to do vae decode
        # import pdb; pdb.set_trace()
        current_time = time.time()
        cvals = self.objective.compute_constraints(seq)
        # print(f"checking candidate for constraints in {time.time() - current_time}")
        # return false if any of cvals are > 0
        if cvals is not None:
            if cvals.max() > 0:
                return False
        return True 


    def sample_random_searchspace_points(self, N):
        lb, ub = self.objective.lb, self.objective.ub 
        if ub is None: ub = self.search_space_data().max() 
        if lb is None: lb = self.search_space_data().max() 
        return torch.rand(N, self.objective.dim)*(ub - lb) + lb


    def randomly_sample_feasible_center(self, higher_ranked_xs, max_n_samples=1_000):
        ''' Rare edge case when we run out of feasible evaluated datapoints
        and must randomly sample to find a new feasible center point
        '''
        # n_samples = 0
        # while True:
        #     if n_samples > max_n_samples:
        #         raise RuntimeError(f'Failed to find a feasible tr center after {n_samples} random samples, recommend use of smaller M or smaller tau')
        #     center_x = self.sample_random_searchspace_points(N=1)
        #     n_samples += 1
        #     if self.is_feasible(center_x, higher_ranked_xs=higher_ranked_xs):
        #         out_dict = self.objective(center_x)
        #         center_score = out_dict['scores'].item() 
        #         # add new point to existing dataset 
        #         self.update_next(
        #             y_next_=torch.tensor(center_score).float(),
        #             x_next_=center_x,
        #         )
        #         break 

        # return center_x, center_x, center_score 
        raise NotImplementedError('Randomly sampling feasible center not yet implemented')


    def generate_batch_single_tr(self, tr_state):
        search_space_cands = generate_batch(
            state=tr_state,
            model=self.model,
            X=self.search_space_data(),
            Y=self.train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
            absolute_bounds=(self.objective.lb, self.objective.ub),
            # considering constraints
            constraint_model_list=self.c_models,
        )

        return search_space_cands


    # def remove_infeasible_candidates(self, x_cands, higher_ranked_cands):
    #     feasible_xs = []
    #     bool_arr = []
    #     print(f"{len(x_cands)} candidates to check")
    #     current_time = time.time()
    #     candidate_counter = 0
    #     for x_cand in x_cands:
    #         test1 = self.is_feasible(x_cand, higher_ranked_cands)
    #         print(f"checking candidate {candidate_counter} in {time.time() - current_time}")
    #         candidate_counter += 1
    #         current_time = time.time()
    #         if test1: # self.is_feasible(x_cand, higher_ranked_cands):
    #             if type(x_cand) is torch.Tensor:
    #                 feasible_xs.append(x_cand.unsqueeze(0))
    #             else:
    #                 feasible_xs.append(x_cand)
    #             bool_arr.append(True)
    #         else:
    #             bool_arr.append(False)
    #     # tracks which were removed vs. kept
    #     bool_arr = np.array(bool_arr)

    #     return feasible_xs, bool_arr

    def is_feasible_diversity(self, x, higher_ranked_xs):
        for higher_ranked_x in higher_ranked_xs:
            if self.objective.divf(x, higher_ranked_x) < self.tau:
                return False
        return True

    def remove_infeasible_candidates(self, x_cands, higher_ranked_cands):
        feasible_xs = []
        bool_arr = []
        # print(f"{len(x_cands)} candidates to check")
        
        # Prepare batch for constraint checking
        if isinstance(x_cands[0], str):
            constraint_batch = x_cands
        else:
            constraint_batch = [x.unsqueeze(0) if isinstance(x, torch.Tensor) else x for x in x_cands]
        
        # Batch constraint check
        current_time = time.time()
        cvals_batch = self.objective.compute_constraints(constraint_batch)
        # print(f"Batch constraint checking completed in {time.time() - current_time} seconds")

        current_time = time.time()
        candidate_counter = 0
        for i, x_cand in enumerate(x_cands):
            # Check constraints
            if cvals_batch is not None and cvals_batch[i].max() > 0:
                bool_arr.append(False)
                continue

            # Check diversity
            if self.is_feasible_diversity(x_cand, higher_ranked_cands):
                if isinstance(x_cand, torch.Tensor):
                    feasible_xs.append(x_cand.unsqueeze(0))
                else:
                    feasible_xs.append(x_cand)
                bool_arr.append(True)
            else:
                bool_arr.append(False)

            # print(f"Checking candidate {candidate_counter} completed in {time.time() - current_time} seconds")
            candidate_counter += 1
            current_time = time.time()

        bool_arr = np.array(bool_arr)
        return feasible_xs, bool_arr


    def update_feasible_candidates_and_tr_state(self, state, feasible_searchspace_pts, feasible_ys, feasible_cs ):
        if len(feasible_ys) > 0:
            if type(feasible_searchspace_pts) is np.ndarray:
                feasible_searchspace_pts = torch.from_numpy(feasible_searchspace_pts).float() 
            if self.minimize:
                feasible_ys = feasible_ys * -1
            self.all_feasible_ys = self.all_feasible_ys + feasible_ys.tolist()
            feasible_searchspace_pts = feasible_searchspace_pts.detach().cpu() 
            self.all_feasible_searchspace_pts = torch.cat((self.all_feasible_searchspace_pts, feasible_searchspace_pts))
            # 4. update state of this tr only on the feasible ys it suggested
            # make sure feasible_ys and feasible_cs are tensors
            feasible_ys = torch.tensor(feasible_ys).float()
            if feasible_cs is not None:
                feasible_cs = torch.tensor(feasible_cs).float()
            update_state(state, feasible_ys, feasible_cs)


    def compute_scores_remaining_cands(self, feasible_searchspace_pts):
        # out_dict = self.objective(torch.cat(feasible_searchspace_pts) )
        # feasible_searchspace_pts=out_dict['valid_zs'] # zs are latent space points
        # feasible_ys=out_dict['scores']
        # feasible_cs=out_dict['constr_vals']
        # feasible_xs=out_dict['valid_xs'] # xs are decoded zs

        # self.all_feasible_xs = self.all_feasible_xs + feasible_xs.tolist()
        # self.all_feasible_cs = self.all_feasible_cs + feasible_cs.tolist()
        
        # return feasible_ys, feasible_searchspace_pts, feasible_cs
        raise NotImplementedError('compute_scores_remaining_cands not implemented')


    def compute_scores_and_update_state(self, state, feasible_searchspace_pts):
        if len(feasible_searchspace_pts) > 0:
            # Compute scores for remaining feasible candiates
            feasible_ys, feasible_searchspace_pts, feasible_cs = self.compute_scores_remaining_cands(feasible_searchspace_pts)
            # Update tr state on feasible candidates 
            self.update_feasible_candidates_and_tr_state(state, feasible_searchspace_pts, feasible_ys, feasible_cs)


    def get_feasible_cands(self, cands):
        # feasible_searchspace_pts, _ = self.remove_infeasible_candidates(
        #     x_cands=cands, 
        #     higher_ranked_cands=self.all_feasible_searchspace_pts
        # )
        # return feasible_searchspace_pts

        raise NotImplementedError('get_feasible_cands not implemented')


    def asymmetric_acquisition(self):   
        '''Generate new candidate points,
        asymetrically discard infeasible ones, 
        evaluate them, and update data
        '''
        # adding constraint support
        if self.train_c is not None: # if constrained 
            constraint_model_list=self.c_models
        else:
            constraint_model_list = None 

        self.all_feasible_xs = [] # (used only by LOL-ROBOT when searchspace pts != xs)
        self.all_feasible_ys = []
        self.all_feasible_cs = []
        self.all_feasible_searchspace_pts = torch.tensor([])
        import time
        current_time = time.time()
        counter = 0

        for state in self.rank_ordered_trs:
            # print(f"starting tr {counter}")
            # 1. Generate a batch of candidates in 
            #   trust region using global surrogate model

            x_next = self.generate_batch_single_tr(state)
            # print(f"generated batch {counter} in {time.time() - current_time}")
            current_time = time.time()

            # 2. Asymetrically remove infeasible candidates
            feasible_searchspace_pts = self.get_feasible_cands(x_next )
            # print(f"got feasible cands {counter} in {time.time() - current_time}")
            current_time = time.time()

            # 3. Compute scores for feassible cands and update tr statee 
            self.compute_scores_and_update_state(state, feasible_searchspace_pts)
            # print(f"computed scores and updated state {counter} in {time.time() - current_time}")
            current_time = time.time()

            counter += 1

        # 4. Add all new evaluated points to dataset (update_next)
        if len(self.all_feasible_searchspace_pts ) != 0:
            self.num_new_points = len(self.all_feasible_ys)
        self.update_data_all_feasible_points() 


    def update_data_all_feasible_points(self):
        # if len(self.all_feasible_searchspace_pts ) != 0:
        #     self.update_next(
        #         y_next_=torch.tensor(self.all_feasible_ys).float(),
        #         x_next_=self.all_feasible_searchspace_pts,
        #     )
        raise NotImplementedError('update_data_all_feasible_points not implemented')


    # def update_next(self, y_next_, x_next_):
    #     '''Add new points (y_next, x_next) to train data
    #     '''
    #     x_next_ = x_next_.detach().cpu() 
    #     if len(x_next_.shape) == 1:
    #         x_next_ = x_next_.unsqueeze(0)
    #     y_next_ = y_next_.detach().cpu()
    #     if len(y_next_.shape) > 1:
    #         y_next_ = y_next_.squeeze() 
    #     #if we imporve 
    #     if y_next_.max() > self.best_score_seen:
    #         self.best_score_seen = y_next_.max().item() 
    #         self.best_x_seen = x_next_[y_next_.argmax()] 
    #     y_next_ = y_next_.unsqueeze(-1)
    #     self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
    #     self.train_x = torch.cat((self.train_x, x_next_), dim=-2)

    def update_next(
        self,
        z_next_,
        y_next_,
        x_next_,
        c_next_=None,
        acquisition=False
    ):
        
        raise NotImplementedError('update_next not implemented')
