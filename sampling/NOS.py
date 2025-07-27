from .base import Algo
import torch
import tqdm
import torch.nn.functional as F
import numpy as np

class NOS(Algo):
    '''
    '''
    def __init__(self, net, data_config,  forward_op=None, num_nos_steps=1, nos_step_size=0.1, nos_stability_coef=0.01, n_max_mutations=None, device='cuda'):
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations, device=device)
        self.nos_step_size = nos_step_size
        self.num_nos_steps = num_nos_steps
        self.nos_stability_coef = nos_stability_coef
        print(f"nos_step_size: {self.nos_step_size}, nos_num_steps: {self.num_nos_steps}, nos_stab_coef: {self.nos_stability_coef}")
        
    def update_model(self, classifier):
        self.forward_op = classifier.eval() #make sure the model is in eval mode

    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)

    def inference(self, num_samples=1, verbose=True, detokenize=False, inpaint=False):
        timesteps = torch.linspace(self.net.timestep-1,1,int((self.net.timestep-1)/1), dtype=int) # iterate over reverse timesteps
        timesteps = tqdm.tqdm(timesteps) if verbose else timesteps
        x = self.net.get_start(num_samples)
        
        for t in timesteps:
            if self.nos_stability_coef is not None:
                x, _, _ = self.net._nos_denoise(classifier_model=self.forward_op, nos_step_size=self.nos_step_size, num_nos_steps=self.num_nos_steps, nos_stability_coef=self.nos_stability_coef, xt=x, t=t)
            else:
                #unguided
                x = self.net.p_sample(x, t, hard=True)

        # project to only the positions under consideration
        # should maybe move this to within the timestep loop but then diversity is really bad
        x = self.project(x)

        # detokenize
        if detokenize:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x]
            #replace special AA characters with a random ones
            #limit the number of mutations
            detokenized = self.project_sequences(detokenized)

            return x, detokenized
        else:
            return x


class NOS_C(Algo):
    """
    Implementation of Classifier Guidance for a continuous diffusion model.
    forward_op is a predictor that outputs log p(y| x_t, t).
    """

    def __init__(self, net, data_config, forward_op=None, temperature=0, num_nos_steps=1, nos_step_size=0.1, nos_stability_coef=0.01, n_max_mutations=None, device='cuda'):
        super().__init__(net=net, forward_op=forward_op, data_config=data_config, n_max_mutations=n_max_mutations,
                         device=device)
        self.nos_step_size = nos_step_size
        self.num_nos_steps = num_nos_steps
        self.nos_stability_coef = nos_stability_coef
        print(f"nos_step_size: {self.nos_step_size}, nos_num_steps: {self.num_nos_steps}, nos_stab_coef: {self.nos_stability_coef}")

    def update_model(self, classifier):
        """ Update the classifier model used in forward_op. """
        self.forward_op = classifier

    def project(self, x):
        return x * self.mask + self.full_seq * (1 - self.mask)

    @torch.no_grad()
    def inference(self, num_samples=1, verbose=False, detokenize=False):
        """
        Generate samples from the continuous diffusion model with classifier guidance.
        We'll assume self.net has a method 'sample_guided(...)' that does the heavy lifting.
        """

        # x = self.net.guided_sample(
        #     forward_op=self.forward_op,
        #     temperature=self.temperature,
        #     mask=self.mask,  # optionally use the mask inside the sampler
        #     num_samples=num_samples,
        #     verbose=verbose
        # ) #fix to be guided_sample

        infill_seed = torch.randint(0, self.net.model.network.vocab_size, (self.seq_len,)).to(
            self.device)  # random seed of token ids for now
        # 1 if != pad, else 0
        infill_mask = (torch.ones(self.seq_len) != self.net.tokenizer.pad_id - 100).to(
            self.device)  # switch 30 for self.net.tokenizer.pad_id

        #  changes infill_mask to only include the residues
        infill_mask = infill_mask * self.mask

        # corrupt_mask: 1 for real tokens, 0 for pad (Equivalent to "fully corrupt all real tokens")
        corrupt_mask = infill_mask.clone().to(self.device)  # (B, L), 1=corrupt, 0=pad

        if self.nos_stability_coef is not None:
            # Define the guidance_kwargs variable
            guidance_kwargs = {
                "guidance_layer": "last",
                "step_size": self.nos_step_size,
                "stability_coef": self.nos_stability_coef,
                "num_steps": self.num_nos_steps
            }

            ##TODO: add support for unconditional sampling
            x = self.net.NOS_C_sample(
                infill_seed=infill_seed,
                infill_mask=infill_mask,
                corrupt_mask=corrupt_mask,
                num_samples=num_samples,
                classifier=self.forward_op, #(should be the actual classifier in this case)  # <--- NEW: The classifier function/nn.Module
                guidance_kwargs=guidance_kwargs,
                bad_word_ids=None,
            )

            # -- 2) (Optionally) force certain positions to remain unchanged after sampling. --
            x = torch.tensor(x) if isinstance(x, np.ndarray) else x
            x = self.project(x.to(device=self.device))
        else:
            x = self.net.guided_sample(
                        infill_seed=infill_seed,
                        infill_mask=infill_mask,
                        corrupt_mask=corrupt_mask,
                        num_samples=num_samples,
                        classifier=None,  # <--- NEW: The classifier function/nn.Module
                        guidance_scale=0.,
                    ) #net.sample doesn't seem to work for some reason
            #convert to torch array of float
            x = torch.tensor(x, dtype=torch.float)
            
        # -- 3) Detokenize if needed. --
        if detokenize:
            # If x is integer tokens:
            detokenized = [self.net.tokenizer.untokenize(s) for s in x]

            detokenized = self.project_sequences(detokenized)
            return x, detokenized
        else:
            return x