import sys
sys.path.append("../")
import fire
import torch 
import pandas as pd
import math 
from robot_scripts.optimize import Optimize
from robot.info_transformer_vae_diverse_objective import InfoTransformerVAEDiverseObjective
import math


class InfoTransformerVAEDiverseOptimization(Optimize):
    """
    Run LOL-ROBOT Optimization using InfoTransformerVAE
    """
    def __init__(
        self,
        dim: int=1024,
        max_string_length: int=100,
        task_specific_args: list=[], # list of additional args to be passed into objective funcion 
        divf_id: str="edit_dist",
        init_data_path: str="../initialization_data/example_init_data.csv",
        **kwargs,
    ):
        self.dim=dim
        self.max_string_length = max_string_length 
        self.task_specific_args = task_specific_args
        self.divf_id = divf_id
        self.init_data_path = init_data_path
        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args['diverseopt'] = locals()
        del self.method_args['diverseopt']['self']

    def initialize_objective(self):
        # initialize objective
        self.objective = InfoTransformerVAEDiverseObjective(
            task_id=self.task_id,
            task_specific_args=self.task_specific_args,
            max_string_length=self.max_string_length,
            dim=self.dim,
            divf_id=self.divf_id,
        )

        return self

    def compute_train_zs(
        self,
        bsz=32,
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

        return init_zs

    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        df = pd.read_csv(self.init_data_path)

        x = df["x"].values.tolist()  
        x = x[0:self.num_initialization_points] 

        y = torch.from_numpy(df["y"].values ).float()
        y = y[0:self.num_initialization_points] 
        y = y.unsqueeze(-1) 

        self.init_train_x = x
        self.init_train_y = y 

        self.init_train_z = self.compute_train_zs()

        return self 


if __name__ == "__main__":
    fire.Fire(InfoTransformerVAEDiverseOptimization)

