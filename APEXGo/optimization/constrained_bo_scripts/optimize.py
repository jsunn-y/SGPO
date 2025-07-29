import torch
import sys 
sys.path.append("../")
import random
import pandas as pd
import numpy as np
import fire
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
os.environ["WANDB_SILENT"] = "True"
from constrained_bo.lol_robot import LolRobotState
from constrained_bo.robot import RobotState
from constrained_bo.latent_space_objective import LatentSpaceObjective
from constrained_bo.objective import Objective
try:
    import wandb
    WANDB_IMPORTED_SUCCESSFULLY = True
except ModuleNotFoundError:
    WANDB_IMPORTED_SUCCESSFULLY = False

torch.set_float32_matmul_precision("highest")

class Optimize(object):
    """
    Run ROBOT Optimization (LOL-Robot used automaticlly for latent space objectives)
    Args:
        M: Number of diverse soltuions desired
        tau: Minimum diversity level required
        task_id: String id for optimization task, by default the wandb project name will be f'optimize-{task_id}'
        seed: Random seed to be set. If None, no particular random seed is set
        track_with_wandb: if True, run progress will be tracked using Weights and Biases API
        wandb_entity: Username for your wandb account (valid username necessary iff track_with_wandb is True)
        wandb_project_name: Name of wandb project where results will be logged (if no name is specified, default project name will be f"APEXGo-{self.task_id}")
        minimize: If True we want to minimize the objective, otherwise we assume we want to maximize the objective
        max_n_oracle_calls: Max number of oracle calls allowed (budget). Optimization run terminates when this budget is exceeded
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: Number evaluated data points used to optimization initialize run
        init_n_update_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        e2e_freq: Number of optimization steps before we update the models end to end (end to end update frequency)
        print_freq: If verbose, program will print out an update every print_freq steps during optimzation. 
        save_csv_frequency: Save all collected data to a csv every save_csv_frequency iterations 
        k: We additionally keep track of and update end to end on the top k points found during optimization
        verbose: If True, we print out updates such as best score found, number of oracle calls made, etc. 
    """
    def __init__(
        self,
        task_id: str="apex",
        M: int=10,
        tau: float=5, 
        seed: int=None,
        track_with_wandb: bool=False,
        wandb_entity: str="",
        wandb_project_name: str="",
        minimize: bool=False,
        max_n_oracle_calls: int=200_000_000,
        learning_rte: float=0.001,
        acq_func: str="ts",
        bsz: int=10,
        num_initialization_points: int=10_000,
        init_n_update_epochs: int=20,
        num_update_epochs: int=2,
        e2e_freq: int=10,
        print_freq: int=10,
        save_csv_frequency: int=1, #changed this to log more frequently
        k: int=1_000,
        verbose: bool=True,
        repeat: int=0, # used to specify which set of initial sequences to use (which random trial)
        **kwargs,
    ):

        # add all local args to method args dict to be logged by wandb
        self.M = M
        self.tau = tau
        self.repeat = repeat
        self.method_args = {}
        self.method_args['init'] = locals()
        del self.method_args['init']['self']
        self.seed = seed
        self.track_with_wandb = track_with_wandb
        self.wandb_entity = wandb_entity 
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.verbose = verbose
        self.num_initialization_points = num_initialization_points
        self.e2e_freq = e2e_freq
        self.print_freq = print_freq
        self.save_csv_frequency = save_csv_frequency
        self.set_seed()
        if wandb_project_name: # if project name specified
            self.wandb_project_name = wandb_project_name
        else: # otherwise use defualt
            self.wandb_project_name = f"APEXGo-{self.task_id}"
        if not WANDB_IMPORTED_SUCCESSFULLY:
            assert not self.track_with_wandb, "Failed to import wandb, to track with wandb, try pip install wandb"
        if self.track_with_wandb:
            assert self.wandb_entity, "Must specify a valid wandb account username (wandb_entity) to run with wandb tracking"

        # initialize latent space objective (self.objective) for particular task
        self.initialize_objective()
        
        # initialize train data for particular task
        #   must define self.init_train_x, self.init_train_y, and if latent obj, self.init_train_z
        self.load_train_data()

        
        if isinstance(self.objective, LatentSpaceObjective): 
            # if we have a latent space objective, use periodic end-to-end updates with the VAE as in LOL-BO (LOL-ROBOT)
            self.lolrobot = True
            RobotStateClass = LolRobotState
        else:
            self.lolrobot = False
            RobotStateClass = RobotState
            self.init_train_z = None
            assert isinstance(self.objective, Objective), "self.objective must be an instance of either LatentSpaceObjective or Objective"

        if self.init_train_c is not None: # if constrained 
            print(f"This is a constrained optimization task")
            assert torch.is_tensor(self.init_train_c), "load_train_data() must set self.init_train_c to a tensor of cs"
            assert self.init_train_c.shape[0] == len(self.init_train_x), f"load_train_data() must initialize exactly the same number of cs and xs, instead got {len(self.init_train_x)} xs and {self.init_train_c.shape[0]} cs"

        # check for correct initialization of train data:
        assert torch.is_tensor(self.init_train_y), "load_train_data() must set self.init_train_y to a tensor of ys"
        assert len(self.init_train_x) == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} xs, instead got {len(self.init_train_x)} xs"
        assert len(self.init_train_y) == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} ys, instead got {len(init_train_y)} ys"
        if self.lolrobot:
            assert torch.is_tensor(self.init_train_z), "load_train_data() must set self.init_train_z to a tensor of zs"
            assert self.init_train_z.shape[0] == self.num_initialization_points, f"load_train_data() must initialize exactly self.num_initialization_points={self.num_initialization_points} zs, instead got {self.init_train_z.shape[0]} zs"
            assert type(self.init_train_x) is list, "load_train_data() must set self.init_train_x to a list of xs"
        else:
            assert torch.is_tensor(self.init_train_x), "load_train_data() must set self.init_train_x to a tensor of xs"
            self.init_train_z is None 

        # TODO: using LolRobotState for now, but should be able to use RobotState
        self.robot_state = LolRobotState(
            M=self.M,
            tau=self.tau,
            objective=self.objective,
            train_x=self.init_train_x,
            train_y=self.init_train_y,
            train_z=self.init_train_z,
            train_c=self.init_train_c,
            minimize=minimize,
            k=k,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_update_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose
        )


    def initialize_objective(self):
        ''' Initialize Objective for specific task
            must define self.objective object
            '''
        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a tensor of x's, or list of x's if latent space objective)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (For latent objectives only,
                    a tensor of corresponding latent space points)
        '''
        return self
    

    def set_seed(self):
        # TODO: check if this is the right way to set seed
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        if self.seed is not None:
            torch.manual_seed(self.seed) 
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(self.seed)

        return self
    

    def create_wandb_tracker(self):
        if self.track_with_wandb:
            config_dict = {k: v for method_dict in self.method_args.values() for k, v in method_dict.items()}
            self.tracker = wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config=config_dict,
            ) 
            self.wandb_run_name = wandb.run.name
        else:
            self.tracker = None 
            self.wandb_run_name = 'no-wandb-tracking'
        
        return self


    def log_data_to_wandb_on_each_loop(self):
        if self.track_with_wandb:
            dict_log = {
                "n_oracle_calls":self.robot_state.objective.num_calls,
                "mean_score_diverse_set":self.robot_state.M_diverse_scores.mean(),
                "min_score_diverse_set":self.robot_state.M_diverse_scores.min(),
                "max_score_diverse_set":self.robot_state.M_diverse_scores.max(),
            }
            for ix, tr_state in enumerate(self.robot_state.rank_ordered_trs):
                rank = ix + 1
                best_x = tr_state.best_x
                if torch.is_tensor(best_x):
                    best_x = best_x.tolist() 
                dict_log[f'diverse_set_X{rank}'] = best_x
                dict_log[f'diverse_set_Y{rank}'] = tr_state.best_value 
                dict_log[f'TR{rank}_length'] = tr_state.length
            self.tracker.log(dict_log)

        if self.step_num % self.save_csv_frequency == 0:
            df = pd.DataFrame()
            save_dir = f'../../../exps/protein/{self.protein}/vae/iterativeBO/APEXGO/'
            #make directories hierarchically if they do not exist
            os.makedirs(save_dir, exist_ok=True)

            file_path = save_dir + f'BO_summary_repeat{self.repeat}.csv'

            df['sequence'] = np.array(self.robot_state.train_x)
            df['fitness'] = self.robot_state.train_y.squeeze().detach().cpu().numpy() 
            #divide by 100 to get the round number
            df['round'] = (df.index/100).astype(int)
            df['repeat'] = self.repeat

            df.to_csv(file_path, index=None)
            print(f"Saved all collected data to {file_path}")

        return self


    def run_robot(self): 
        ''' Main optimization loop
        '''
        self.step_num = 0
        # creates wandb tracker iff self.track_with_wandb == True
        self.create_wandb_tracker()
        # log init data
        self.log_data_to_wandb_on_each_loop()
        #main optimization loop 
        import time
        while self.robot_state.objective.num_calls < self.max_n_oracle_calls:
            print(f"\nStep {self.step_num}")
            # update models end to end when we fail to make
            #   progress e2e_freq times in a row (e2e_freq=10 by default)
            if self.lolrobot and (self.robot_state.progress_fails_since_last_e2e >= self.e2e_freq):
                print(f"Updating models end to end")
                self.robot_state.update_models_e2e()
            else: # otherwise, just update the surrogate model on data
                print(f"Updating surrogate model")
                self.robot_state.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.robot_state.asymmetric_acquisition()
            # check if restart is triggered for any individual tr and restart it as needed
            print(f"Restarting TRs as needed")
            self.robot_state.restart_trs_as_needed() 
            # recenter trust regions to maintain feasible set
            print(f"Recentering TRs")
            self.robot_state.recenter_trs() 
            self.step_num += 1
            # log best feassible set found to wandb
            print(f"Logging data to wandb")
            self.log_data_to_wandb_on_each_loop()
            # periodically print updates
            if self.verbose and (self.step_num % self.print_freq == 0):
                self.print_progress_update()

        # if verbose, print final results
        if self.verbose:
            print("\nOptimization Run Finished, Final Results:")
            self.print_progress_update()

        # log final set of M diverse solutions found as a table
        self.log_final_table_wandb()

        return self 


    def print_progress_update(self):
        ''' Important data printed each time a new
            best input is found, as well as at the end 
            of the optimization run
            (only used if self.verbose==True)
            More print statements can be added her as desired
        '''
        if self.track_with_wandb:
            print(f"\nOptimization Run: {self.wandb_project_name}, {wandb.run.name}")
        print(f"{self.step_num} Steps Complete")
        print(f"Total Number of Oracle Calls (Function Evaluations): {self.robot_state.objective.num_calls}")
        print(f"Diverse (divf >= {self.tau}) set of {self.M} solultions have been found with...") 
        print(f"    Mean {self.task_id} Score = {self.robot_state.M_diverse_scores.mean()}") 
        print(f"    Min {self.task_id} Score = {self.robot_state.M_diverse_scores.min()}")
        print(f"    Max {self.task_id} Score = {self.robot_state.M_diverse_scores.max()}")

        return self


    def log_final_table_wandb(self):
        ''' After optimization finishes, log final
            set of M diverse solutions as a table
        '''
        if self.track_with_wandb:
            cols = ["Final Diverse Solutions", "Objective Values"]
            data_list = []
            for ix, solution in enumerate(self.robot_state.M_diverse_xs):
                if type(solution) is np.ndarray or torch.is_tensor(solution):
                    solution = solution.tolist() 
                data_list.append([ solution, self.robot_state.M_diverse_scores[ix] ]) 
            final_diverse_set = wandb.Table(columns=cols, data=data_list)
            self.tracker.log({"final_diverse_set_table": final_diverse_set})
            self.tracker.finish()

        return self


    def done(self):
        return None


def new(**kwargs):
    return Optimize(**kwargs)

if __name__ == "__main__":
    fire.Fire(Optimize)
