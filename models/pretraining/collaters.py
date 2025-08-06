# Modified from the NOS-C noise scheduler: https://github.com/ngruver/NOS/blob/569f1c85adf2ca2ea5e32efdf46276995fcf322c/seq_models/schedule/noise_schedule.py#L114 #

import torch
import numpy as np
from evodiff.utils import Tokenizer
from sequence_models.constants import PAD, PROTEIN_ALPHABET


import enum
import math
import copy

import torch
import torch.nn as nn

import transformers

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == "sd-10": # hoogeboom
        d = 10.0  # or another value for different corruption rate
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 / (1 + d ** 2 * math.tan((math.pi * t / 2)) ** 2)
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start =  scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar2(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusionSchedule:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            timesteps,
            noise_schedule,
            noise_scale=1.0,
    ):
        betas = get_named_beta_schedule(noise_schedule, timesteps)

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.timesteps = int(betas.shape[0])
        self.noise_scale = noise_scale

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )
        self.sigmas = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod/self.sqrt_alphas_cumprod)
        
        
    def sigma_inv(self, sigma):
        # Find the index of the closest value in self.sigmas to the given sigma
        idx = (torch.abs(self.sigmas - sigma)).argmin()
        return idx

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = self.noise_scale * torch.randn_like(x_start)
            # add scaling here
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        posterior_variance = (self.noise_scale ** 2) * posterior_variance
        posterior_log_variance_clipped = 2 * np.log(self.noise_scale) + posterior_log_variance_clipped

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def _pad(tokenized, value, max_len=None):
    """
    Pads a list of 1D LongTensors to a specified max length.
    If max_len is not provided, defaults to the longest sequence in the batch.
    Returns a (batch_size, max_len) LongTensor.
    """
    batch_size = len(tokenized)
    # Determine max_len dynamically if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in tokenized)

    # Create a tensor filled with the pad value
    output = torch.full((batch_size, max_len), value, dtype=torch.long)

    # Populate the tensor with sequences, truncating if necessary
    for i, seq in enumerate(tokenized):
        seq_len = min(len(seq), max_len)
        output[i, :seq_len] = seq[:seq_len]

    return output

# def _pad(tokenized, value):
#     """
#     Utility function that pads a list of 1D LongTensors to the same length.
#     Returns a (batch_size, max_len) LongTensor.
#     """
#     batch_size = len(tokenized)
#     max_len = max(len(seq) for seq in tokenized)
#     output = torch.full((batch_size, max_len), value, dtype=torch.long)
#     for i, seq in enumerate(tokenized):
#         output[i, :len(seq)] = seq
#     return output

class ContinuousCollater(object):
    """
    A simplified collater for continuous (Gaussian) diffusion that:
      Tokenizes each sequence into integer IDs.
      Pads them to same length.
      Creates:
         - attn_mask: 1 = real token, 0 = pad
         - corrupt_mask: 1 = "noise this token", 0 = "keep token"
           (corrupt_mask=1 for all real tokens, matching the discrete
            collater's logic of "fully corrupt" by default)
      Returns a dictionary consistent with continuous model's forward function:
         {
           "seq": (B, L) int IDs,
           "attn_mask": (B, L) 1=real,0=pad,
           "corrupt_mask": (B, L) 1=noise,0=keep,
           "labels": None   # for regression labels, attach them here
         }
    """

    def __init__(self, tokenizer, max_len):
        """
        Args:
            tokenizer: An EvoDiff-like Tokenizer with:
                       - tokenize(seq_str) -> List[int]
                       - pad_id for padding
                       - max_len for truncation/padding
        """
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.max_len = max_len # hard coded for now, this is necessary for shape matching in continuous model

    def __call__(self, sequences):
        """
        `sequences` is a list of tuples: [(seq_str,), (seq_str,), ...]
        coming from SequenceDataset.
        """
        # 1) Tokenize sequences and handle empty ones
        tokenized = []
        del_indices = []
        for i, seq_tuple in enumerate(sequences):
            #seq_str = seq_tuple[0]
            ids = self.tokenizer.tokenize(seq_tuple)
            if len(ids) == 0:
                # Remove empty sequences from the batch
                del_indices.append(i)
                continue
            tokenized.append(torch.tensor(ids, dtype=torch.long))

        for idx in reversed(del_indices):
            sequences.pop(idx)

        # if all are empty return empty batch
        if len(tokenized) == 0:
            return {
                "seq": torch.empty(0, dtype=torch.long),
                "attn_mask": torch.empty(0, dtype=torch.long),
                "corrupt_mask": torch.empty(0, dtype=torch.long),
                "labels": None
            }

        # pad to same length
        padded = _pad(tokenized, value=self.pad_id, max_len=self.max_len)  # shape (B, L)
        batch_size, max_len = padded.shape

        # attn_mask: 1 if != pad, else 0
        attn_mask = (padded != self.pad_id).long()

        # corrupt_mask: 1 for real tokens, 0 for pad (Equivalent to "fully corrupt all real tokens")
        corrupt_mask = attn_mask.clone()  # (B, L), 1=corrupt, 0=pad

        # return dictionary (no timesteps because the model samples them, might change later)
        #    no one-hot, no Q, etc. since model will do the actual noising
        return {
            "seq": padded,              # (B, L) integer tokens
            "attn_mask": attn_mask,     # (B, L) binary
            "corrupt_mask": corrupt_mask, # (B, L) binary
            "labels": None              # or real labels if you have them
        }

import torch
import torch
import esm
import numpy as np

class ESMTokenizer:
    def __init__(self, esm_model_name='esm2_t12_35M_UR50D', sequences=True):
        self.model, self.alphabet = getattr(esm.pretrained, esm_model_name)()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.padding_idx = self.alphabet.padding_idx

    @property
    def pad_id(self):
        return self.padding_idx


    def tokenize(self, seq):
        # Match EvoDiff: accepts a tuple like (sequence,)
        if isinstance(seq, tuple):
            seq = seq[0]
        return np.array(self.alphabet.encode(seq))

    def untokenize(self, tokenized_seq):
        # tokenized_seq is a tensor or numpy array
        if torch.is_tensor(tokenized_seq):
            tokenized_seq = tokenized_seq.cpu().numpy()
        tokens = [self.alphabet.get_tok(int(tok)) for tok in tokenized_seq if tok != self.padding_idx]
        # Remove special tokens from untokenized output
        tokens = [tok for tok in tokens if tok not in ('<cls>', '<eos>', '<pad>')]
        return "".join(tokens)

import random
from typing import Any, Dict, List, Set, Tuple, Union
from tokenizers import Tokenizer, Encoding

class CausalCollater(object):
    """
    A simplified collater for the ProGen2 Causal LM models.
      Tokenizes each sequence into integer IDs.
      Pads them to same length.
    Adapted from https://github.com/Profluent-Internships/ProCALM/blob/main/progen_conditional/data/prepare.py
    """

    def __init__(self, tokenizer, reverse=False, paired_mode=False): # max_len
        """
        Args:
            tokenizer: A ProGen2 Tokenizer
            reverse: whether or not to reverse sequences during training, usually reverse during pretraining but not during DPO
            paired_mode: whether or not to use paired mode (only applies to DPO)
        """
        self.reverse = reverse #whether or not to reverse sequences during training
        self.tokenizer = tokenizer
        self.paired_mode = paired_mode
        #self.max_len = max_len + 2 #for start and stop tokens
    
    def construct_padded_tensors(self, max_length: int, encs: List[Encoding]):
        attention_mask = np.zeros((len(encs), max_length), dtype=bool)
        input_ids = np.zeros((len(encs), max_length), dtype=int)
        
        for i, enc in enumerate(encs):
            assert len(enc.attention_mask) == len(enc.ids)

            attention_mask[i, : len(enc.attention_mask)] = enc.attention_mask
            input_ids[i, : len(enc.ids)] = enc.ids

        return input_ids, attention_mask

    def __call__(self, batch):
        """
        `sequences` is a list of tuples: [(seq_str,), (seq_str,), ...]
        coming from SequenceDataset or ProteinPredictorDataset
        """
        try:
            seqs, y = zip(*batch) #unzip y values when provided
        except ValueError:
            seqs = batch
            y = None 

        seqs = [seq[0] for seq in seqs]
        processed_seqs: List[str] = []
        if self.reverse:
            reverse_booleans = [random.choice([0, 1]) for _ in range(len(seqs))]
        else:
            reverse_booleans = [False] * len(seqs) #don't reverse any sequences
        
        for seq, reverse in zip(seqs, reverse_booleans):
            #randomly reverse seq based on rval probability
            seq = '1' + seq + '2'
            if reverse:
                seq = seq[::-1]
            processed_seqs.append(seq)
        
        seq_encodings = self.tokenizer.encode_batch(processed_seqs)

        #max_length = len(seq_encodings[-1].ids)
        lengths = [len(enc.attention_mask) for enc in seq_encodings]
        max_length = max(lengths)

        #Construct padded attention masks, position ids and sequence ids
        input_ids, attention_mask = self.construct_padded_tensors(max_length, seq_encodings)

        batch = dict(
            input_ids=torch.tensor(input_ids),
            labels=torch.tensor(input_ids),
            attention_mask=torch.tensor(attention_mask, dtype=torch.bool),
            sequences=seqs, #doesn't seem to increase memory
        )
        if y is not None:
            y = torch.tensor(y, dtype=torch.float32)
            batch["weights"] = y

        return batch

class PredictorCollater(object):
    """
    A simplified collater to tokenize sequences for the ProteinPredictor.
    """

    def __init__(self, tokenizer): # max_len
        """
        Args:
            tokenizer: A diffussion model tokenizer from evodiff
        """
        self.tokenizer = tokenizer

    def encode_seqs(self, seqs):
        x = []
        for seq in seqs:
            data = [self.tokenizer.tokenize(s) for s in seq]
            x.append(torch.from_numpy(np.array(data)))
        x = torch.stack(x).squeeze(-1)
        return x

    def __call__(self, batch):
        """
        `sequences` is a list of tuples: [(seq_str,), (seq_str,), ...]
        coming from ProteinPredictorDataset
        """
        seqs, y = zip(*batch)
        seqs = [seq[0] for seq in seqs]
        x = self.encode_seqs(seqs)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

class MDLMCollater(object):
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.K = self.tokenizer.K
        self.pad_id = self.tokenizer.pad_id
        
    def __call__(self, sequences):
        tokenized = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        max_len = max(len(t) for t in tokenized)
        padded = torch.full((len(tokenized), max_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(tokenized):
            padded[i, :len(seq)] = seq
        return padded

collate_fn_mapping = {"cls_guidance": PredictorCollater, "DPO": CausalCollater}