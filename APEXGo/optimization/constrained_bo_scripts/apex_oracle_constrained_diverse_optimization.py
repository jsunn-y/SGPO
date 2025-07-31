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
from Bio import SeqIO


class APEXConstrainedDiverseOptimization(Optimize):
    """
    Run LOL-ROBOT Constrained Optimization using InfoTransformerVAE
    """
    def __init__(
        self,
        dim: int=256, # SELFIES VAE DEFAULT LATENT SPACE DIM
        path_to_vae_statedict: str=None,
        max_string_length: int=50,
        task_specific_args: list=[], # list of additional args to be passed into objective funcion 
        constraint_function_ids: list=[], # list of strings identifying the black box constraint function to use
        constraint_thresholds: list=[], # list of corresponding threshold values (floats)
        constraint_types: list=[], # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
        divf_id: str="edit_dist",
        # init_data_path: str=None,
        repeat: int=0, # used to specify which set of initial sequences to use (which random trial)
        **kwargs,
    ):
        self.dim=dim
        self.repeat = repeat
        self.max_string_length = max_string_length 
        self.task_specific_args = task_specific_args
        self.protein = task_specific_args[0] # protein name to be used in oracle inference
        self.divf_id = divf_id
        # TODO: We currently are hard coding the init data path
        # self.init_data_path = init_data_path
        
        self.path_to_init_seqs = f"../../../exps/protein/{self.protein}/initial_sample_d3pm/unconstrained/"

        print("task_specific_args: ", task_specific_args)
        print("constraint_function_ids: ", constraint_function_ids)
        print("constraint_thresholds: ", constraint_thresholds)
        print("constraint_types: ", constraint_types)

        self.protein = task_specific_args[0]

        if self.protein == "CreiLOV":
            self.path_to_vae_statedict = "/disk1/jyang4/repos/APEXGo/generation/saved_models/exalted-pine-5-CreiLOV/exalted-pine-5_model_state_epoch_241.pkl"
        elif self.protein == "TrpB":
            self.path_to_vae_statedict = "/disk1/jyang4/repos/APEXGo/generation/saved_models/dainty-dream-9-TrpB/dainty-dream-9_model_state_epoch_421.pkl"
        elif self.protein == "GB1":
            self.path_to_vae_statedict = "/disk1/jyang4/repos/APEXGo/generation/saved_models/absurd-cosmos-6-GB1/absurd-cosmos-6_model_state_epoch_391.pkl"
        else:
            raise ValueError(f"Unsupported protein: {self.protein}. Please specify a valid protein name.")


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
            protein =self.protein, # protein name to be used in oracle inference
        )

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
        
        #for now, use the first 100 sequences from the d3pm initial round
        train_x_seqs = [str(record.seq) for record in SeqIO.parse(self.path_to_init_seqs + f"generated_{self.repeat}.fasta", "fasta")][:self.num_initialization_points]

        # Use oracle to get initial scores
        self.num_initialization_points = min(self.num_initialization_points, len(train_x_seqs))
        self.load_train_z()
        self.init_train_x = train_x_seqs[0:self.num_initialization_points]
        
        # Initialize objective if not already done
        if not hasattr(self, 'objective'):
            self.initialize_objective()
        
        # Get scores from oracle
        #print(self.init_train_x)
        train_y = self.objective.objective_function(self.init_train_x)
        self.init_train_y = torch.tensor(train_y).unsqueeze(-1)

        #Moved this from initialize_objective
        # if train zs have not been pre-computed for particular vae, compute them 
        #   by passing initialization selfies through vae 
        if self.init_train_z is None:
            self.init_train_z = self.compute_train_zs()

        self.init_train_c = self.objective.compute_constraints(self.init_train_x)

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

