import numpy as np
import torch 
import sys 
import random
sys.path.append("../")
from uniref_vae.data import collate_fn
from uniref_vae.load_vae import load_vae
from constrained_bo.latent_space_objective import LatentSpaceObjective
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT 
from your_tasks.your_diversity_functions import DIVERSITY_FUNCTIONS_DICT 
from your_tasks.your_blackbox_constraints import CONSTRAINT_FUNCTIONS_DICT 

from constants import (
    PATH_TO_VAE_STATE_DICT,
    ALL_AMINO_ACIDS
)
import math 

class ApexConstrainedDiverseObjective(LatentSpaceObjective):
    '''Objective class supports all optimization tasks using the 
        InfoTransformerVAE '''

    def __init__(
        self,
        task_id='apex', # id of objective funciton you want to maximize 
        task_specific_args=[],
        divf_id="edit_dist",
        path_to_vae_statedict=PATH_TO_VAE_STATE_DICT,
        dim=256, # SELFIES VAE DEFAULT LATENT SPACE DIM
        xs_to_scores_dict={},
        max_string_length=50,
        num_calls=0,
        lb=None,
        ub=None,
        constraint_function_ids=[], # list of strings identifying the black box constraint function to use
        constraint_thresholds=[], # list of corresponding threshold values (floats)
        constraint_types=[], # list of strings giving correspoding type for each threshold ("min" or "max" allowed)
    ):
        self.dim                    = dim # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.task_specific_args     = task_specific_args
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.divf_id                = divf_id # specify which diversity function to use with string id 
        assert task_id in OBJECTIVE_FUNCTIONS_DICT 
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](*self.task_specific_args)

        self.constraint_functions       = []
        for ix, constraint_threshold in enumerate(constraint_thresholds):
            cfunc_class = CONSTRAINT_FUNCTIONS_DICT[constraint_function_ids[ix]]
            cfunc = cfunc_class(
                threshold_value=constraint_threshold,
                threshold_type=constraint_types[ix],
            )
            self.constraint_functions.append(cfunc) 
        
        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
            dim=self.dim, #  DEFAULT VAE LATENT SPACE DIM
            lb=lb,
            ub=ub,
        )

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        # sample peptide string form VAE decoder
        # import pdb; pdb.set_trace()
        sample = self.vae.sample(z=z.reshape(-1, 2, self.dim//2))
        # grab decoded aa strings
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        # get rid of X's (deletion)
        temp = [] 
        for seq in decoded_seqs:
            seq = seq.replace("X", "A")
            seq = seq.replace("-", "")
            if len(seq) == 0:
                seq = "AAA" # catch empty string case too... 
            temp.append(seq)
        decoded_seqs = temp

        return decoded_seqs


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # TODO: make this consistent with lolbo implementation
        # This is different from lolbo implementation since we call it one at a time
        scores_list = self.objective_function([x])
        return scores_list[0]


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.vae, self.dataobj = load_vae(
            path_to_vae_statedict=self.path_to_vae_statedict,
            dim=self.dim,
            max_string_length=self.max_string_length,
        )

        # make sure max string length is set correctly
        print("max string length: ", self.vae.max_string_length)
        # flush
        sys.stdout.flush()


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss


    def divf(self, x1, x2):
        ''' Compute diversity function between two 
            potential xs/ sequences so we can
            create a diverse set of optimal solutions
            with some minimum diversity between eachother
        '''
        return DIVERSITY_FUNCTIONS_DICT[self.divf_id](x1, x2) 
    
    @torch.no_grad()
    def compute_constraints(self, xs_batch):
        ''' Input: 
                a list xs (list of sequences)
            Output: 
                c: tensor of size (len(xs),n_constraints) of
                    resultant constraint values, or
                    None of problem is unconstrained
                    Note: constraints, must be of form c(x) <= 0!
        '''
        if len(self.constraint_functions) == 0:
            return None 
        
        all_cvals = []
        for cfunc in self.constraint_functions:
            cvals = cfunc(xs_batch)
            all_cvals.append(cvals)

        return torch.cat(all_cvals, -1)
