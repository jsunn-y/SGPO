import numpy as np
import torch
torch.set_float32_matmul_precision("highest")
from constrained_bo.objective import Objective


class LatentSpaceObjective(Objective):
    '''Base class for any latent space optimization task
        class supports any optimization task with accompanying VAE
        such that during optimization, latent space points (z) 
        must be passed through the VAE decoder to obtain 
        original input space points (x) which can then 
        be passed into the oracle to obtain objective values (y)
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
        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id=task_id,
            dim=dim,
            lb=lb,
            ub=ub,
        )
        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None


    def __call__(self, z, decoded_xs=None):
        ''' Input 
                z: a numpy array or pytorch tensor of latent space points
                decoded_xs: option to pass in list of decoded xs for efficiency if the zs have already been decoded
            Output
                out_dict['valid_zs'] = the zs which decoded to valid xs 
                out_dict['valid_xs'] = an array of valid xs obtained from input zs
                out_dict['scores']: an array of valid scores obtained from input zs
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        # if no decoded xs passed in, we decode the zs to get xs
        if decoded_xs is None: 
            decoded_xs = self.vae_decode(z)

        out_dict = self.xs_to_valid_scores(decoded_xs)
        valid_zs = z[out_dict['bool_arr']] 
        out_dict['valid_zs'] = valid_zs
        # get valid constraint values for valid decoded xs
        out_dict['constr_vals'] = self.compute_constraints(out_dict['valid_xs'])

        return out_dict


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        raise NotImplementedError("Must implement vae_decode()")


    def initialize_vae(self):
        ''' Sets variable self.vae to the desired pretrained vae '''
        raise NotImplementedError("Must implement method initialize_vae() to load in vae for desired optimization task")


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
        raise NotImplementedError("Must implement method vae_forward() (forward pass of vae)")

    def compute_constraints(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                c: tensor of size (len(xs),n_constraints) of
                    resultant constraint values, or
                    None of problem is unconstrained
                    Note: constraints, must be of form c(x) <= 0!
        '''
        return None 