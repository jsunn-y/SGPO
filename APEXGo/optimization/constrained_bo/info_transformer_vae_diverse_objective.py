import numpy as np
import torch 
import sys 
sys.path.append("../")
from robot.latent_space_objective import LatentSpaceObjective
from uniref_vae.data import collate_fn 
from uniref_vae.load_vae import load_vae
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT 
from your_tasks.your_diversity_functions import DIVERSITY_FUNCTIONS_DICT 


class InfoTransformerVAEDiverseObjective(LatentSpaceObjective):
    '''Objective class supports all optimization tasks using the 
        InfoTransformerVAE '''

    def __init__(
        self,
        task_id='example', # id of objective funciton you want to maximize 
        task_specific_args=[],
        divf_id="edit_dist",
        path_to_vae_statedict="../uniref_vae/saved_models/dim512_k1_kl0001_acc94_vivid-cherry-17_model_state_newest.pkl",
        dim=1024,
        xs_to_scores_dict={},
        max_string_length=50,
        num_calls=0,
        lb=None,
        ub=None,
    ):
        self.dim                    = dim # VAE latent space dim 
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.task_specific_args     = task_specific_args
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.divf_id                = divf_id # specify which diversity function to use with string id 
        assert task_id in OBJECTIVE_FUNCTIONS_DICT 
        self.objective_function = OBJECTIVE_FUNCTIONS_DICT[task_id](*self.task_specific_args)
        
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
                z: a tensor latent space points (bsz, self.dim)
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
        sample = self.vae.sample(z=z.reshape(-1, 2, self.dim//2))
        # grab decoded aa strings
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        return decoded_seqs


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
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
