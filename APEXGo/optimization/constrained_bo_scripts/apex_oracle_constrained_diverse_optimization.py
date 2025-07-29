import sys
sys.path.append("../")
import fire
import torch 
import pandas as pd
import math 
from constrained_bo_scripts.optimize import Optimize
from constrained_bo.apex_oracle_constrained_diverse_objective import ApexConstrainedDiverseObjective
import math
from constants import (
    PATH_TO_VAE_STATE_DICT,
)
torch.set_num_threads(1)

class APEXConstrainedDiverseOptimization(Optimize):
    """
    Run LOL-ROBOT Constrained Optimization using InfoTransformerVAE
    """
    def __init__(
        self,
        dim: int=256, # SELFIES VAE DEFAULT LATENT SPACE DIM
        path_to_vae_statedict: str=PATH_TO_VAE_STATE_DICT,
        max_string_length: int=50,
        task_specific_args: list=[], # list of additional args to be passed into objective funcion 
        constraint_function_ids: list=[], # list of strings identifying the black box constraint function to use
        constraint_thresholds: list=[], # list of corresponding threshold values (floats)
        constraint_types: list=[], # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        divf_id: str="edit_dist",
        # init_data_path: str=None,
        **kwargs,
    ):
        self.dim=dim
        self.path_to_vae_statedict = path_to_vae_statedict
        self.max_string_length = max_string_length 
        self.task_specific_args = task_specific_args
        self.divf_id = divf_id
        # TODO: We currently are hard coding the init data path
        # self.init_data_path = init_data_path

        print("task_specific_args: ", task_specific_args)
        print("constraint_function_ids: ", constraint_function_ids)
        print("constraint_thresholds: ", constraint_thresholds)
        print("constraint_types: ", constraint_types)

        self.score_version = task_specific_args[0]

        assert len(constraint_function_ids) == len(constraint_thresholds)
        assert len(constraint_thresholds) == len(constraint_types)
        self.constraint_function_ids = constraint_function_ids # list of strings identifying the black box constraint function to use
        self.constraint_thresholds = constraint_thresholds # list of corresponding threshold values (floats)
        self.constraint_types = constraint_types # list of strings giving correspoding type for each threshold ("min" or "max" allowed)

        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args['diverseopt'] = locals()
        del self.method_args['diverseopt']['self']

    def initialize_objective(self):
        # initialize objective
        self.objective = ApexConstrainedDiverseObjective(
            task_id=self.task_id,
            task_specific_args=self.task_specific_args,
            path_to_vae_statedict=self.path_to_vae_statedict,
            max_string_length=self.max_string_length,
            dim=self.dim,
            divf_id=self.divf_id,
            constraint_function_ids=self.constraint_function_ids, # list of strings identifying the black box constraint function to use
            constraint_thresholds=self.constraint_thresholds, # list of corresponding threshold values (floats)
            constraint_types=self.constraint_types, # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        )

        # if train zs have not been pre-computed for particular vae, compute them 
        #   by passing initialization selfies through vae 
        if self.init_train_z is None:
            self.init_train_z = self.compute_train_zs()
        self.init_train_c = self.objective.compute_constraints(self.init_train_x)

        return self

    def compute_train_zs(
        self,
        bsz=64
    ):
        init_zs = []
        # make sure vae is in eval mode 
        self.objective.vae.eval() 
        n_batches = math.ceil(len(self.init_train_x)/bsz)
        for i in range(n_batches):
            xs_batch = self.init_train_x[i*bsz:(i+1)*bsz] 
            zs, _ = self.objective.vae_forward(xs_batch)
            init_zs.append(zs.detach().cpu())
        init_zs = torch.cat(init_zs, dim=0)
        # now save the zs so we don't have to recompute them in the future:
        state_dict_file_type = self.objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
        path_to_init_train_zs = self.objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
        zs_arr = init_zs.cpu().detach().numpy()
        pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 

        return init_zs

    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        filename_seqs = f"../apex_oracle/init_data/init_seqs.csv"
        df = pd.read_csv(filename_seqs, header=None)
        train_x_seqs = df.values.squeeze().tolist()
        filename_scores = f"../apex_oracle/init_data/{self.score_version}_scores.csv"
        df = pd.read_csv(filename_scores, header=None)
        train_y = torch.from_numpy(df.values).float()
        
        self.num_initialization_points = min(self.num_initialization_points, len(train_x_seqs))
        self.load_train_z()
        self.init_train_x = train_x_seqs[0:self.num_initialization_points]
        train_y = train_y[0:self.num_initialization_points]
        self.init_train_y = train_y #.unsqueeze(-1)
        return self 
    
    def load_train_z(
        self,
    ):
        state_dict_file_type = self.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
        path_to_init_train_zs = self.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
        # if we have a path to pre-computed train zs for vae, load them
        try:
            zs = pd.read_csv(path_to_init_train_zs, header=None).values
            # make sure we have a sufficient number of saved train zs
            assert len(zs) >= self.num_initialization_points
            zs = zs[0:self.num_initialization_points]
            zs = torch.from_numpy(zs).float()
        # otherwisee, set zs to None 
        except: 
            zs = None 
        self.init_train_z = zs 
        return self


if __name__ == "__main__":
    fire.Fire(APEXConstrainedDiverseOptimization)

