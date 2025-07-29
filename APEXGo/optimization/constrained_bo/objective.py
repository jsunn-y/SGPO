import numpy as np
import torch 

class Objective:
    '''Base class for any optimization task
        class supports oracle calls and tracks
        the total number of oracle class made during 
        optimization 
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        task_id='',
        dim=256,
        lb=None,
        ub=None,
    ):
        # dict used to track xs and scores (ys) queried during optimization
        self.xs_to_scores_dict = xs_to_scores_dict 
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id
        # latent dimension of vaae
        self.dim = dim
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub  


    def __call__(self, xs):
        ''' Input 
                x: a numpy array or pytorch tensor of search space points
            Output
                out_dict['valid_xs'] = an array of xs passed in from which valid scores were obtained 
                out_dict['scores']: an array of valid scores obtained from input xs
        '''
        if type(xs) is np.ndarray: 
            xs = torch.from_numpy(xs).float()
        out_dict = self.xs_to_valid_scores(xs)
        return out_dict


    def xs_to_valid_scores(self, xs):
        scores = []
        for idx, x in enumerate(xs):
            # if we have already computed the score, don't 
            #   re-compute (don't call oracle unnecessarily)
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
            else: # otherwise call the oracle to get score
                score = self.query_oracle(x)
                # add score to dict so we don't have to
                #   compute it again if we get the same input x
                self.xs_to_scores_dict[x] = score
                # track number of oracle calls 
                #   nan scores happen when we pass an invalid
                #   peptide string and thus avoid calling the
                #   oracle entirely
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1
            scores.append(score)
        scores_arr = np.array(scores)
        if type(xs) is list: 
            xs = np.array(xs) 
        elif type(xs) is torch.Tensor:
            xs = xs.detach().cpu().numpy() 
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr)) 
        valid_xs = xs[bool_arr]
        valid_scores = scores_arr[bool_arr]
        out_dict = {}
        out_dict['scores'] = valid_scores
        out_dict['bool_arr'] = bool_arr
        out_dict['valid_xs'] = valid_xs

        return out_dict


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")


    def divf(self, x1, x2):
        ''' Input: 
                x1 and x2, two arbitrary items from search space X
            Output: 
                float giving some measure of diversity between x1 and x2
        '''
        raise NotImplementedError("Must implement method divf() (diversity function)")
