import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import hydra

from evodiff.pretrained import D3PM_UNIFORM_38M
from evodiff.collaters import D3PMCollater
from models.pretraining.collaters import ContinuousCollater, CausalCollater, MDLMCollater
from models.pretraining.model.d3pm_evodiff import pretrained_mapping

from models.pretraining.model.progen2.tokenizer import get_tokenizer #for language models

class SequenceDataset(Dataset):
    """
    For pretraining or finetuning prior models. Loads protein sequences of an MSA from a processed csv file with splits.
    """
    def __init__(self, config, split):
        super().__init__()
        self.config = config
        dataset_path = config.data.pretrain_data_prefix + config.pretrain_model.data.dataset_suffix + ".csv" #format of the pretraining data is prefix and suffix separated by "_" and ".csv" as the extension

        df = pd.read_csv(dataset_path)

        df = df[df['Split'] == split]
        # df = df.head(64) #for testing
        self.seqs = df['sequence'].tolist()
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index) -> str:
        return (self.seqs[index],) #to be consistent with sequence_models.datasets.UniRefDataset from https://github.com/microsoft/protein-sequence-models/blob/main/sequence_models/datasets.py

def get_loaders(config):
    if "d3pm" in config.pretrain_model.name:
        if 'finetune' in config.pretrain_model.name:
            network, _, tokenizer, _ = pretrained_mapping[config.pretrain_model.model.model_config.pretrained_evodiff_ckpt](return_all=False)
            diffusion_timesteps = network.embedder.timesteps
        else:
            # Discrete: timesteps is stored in the 'network' section
            diffusion_timesteps = config.pretrain_model.model.network.timesteps
            if config.pretrain_model.model.model_config.mask == 'uniform':
                tokenizer = hydra.utils.instantiate(config.pretrain_model.model.tokenizer, sequences=True)

        if config.pretrain_model.model.model_config.mask == 'uniform':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        # Currently not supported
        # if config.mask == 'blosum':
        #     Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        collater = D3PMCollater(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)
    elif "continuous_ESM" in config.pretrain_model.name:
        diffusion_timesteps = config.pretrain_model.model.noise_schedule.timesteps
        tokenizer = hydra.utils.instantiate(config.pretrain_model.model.tokenizer, sequences=True)
        collater = ContinuousCollater(tokenizer=tokenizer, max_len=config.data.seq_len)
    elif "continuous" in config.pretrain_model.name:
        # Continuous: timesteps is stored in the noise_schedule section
        diffusion_timesteps = config.pretrain_model.model.noise_schedule.timesteps
        tokenizer = hydra.utils.instantiate(config.pretrain_model.model.tokenizer, sequences=True)
        collater = ContinuousCollater(tokenizer=tokenizer, max_len=config.data.seq_len)
    elif "causal" in config.pretrain_model.name:
        tokenizer = get_tokenizer()
        collater = CausalCollater(tokenizer=tokenizer, reverse=True) #randomly reverse some of the sequences can improve training #max_len=config.data.seq_len
    elif "mdlm" in config.pretrain_model.name or "udlm" in config.pretrain_model.name:
        tokenizer = hydra.utils.instantiate(config.pretrain_model.model.tokenizer, sequences=True)
        collater = MDLMCollater(tokenizer=tokenizer)
           
    dsets = [SequenceDataset(config, split) for split in config.pretrain_model.data.splits]

    effective_batch_size = config.pretrain_model.train.batch_size
    if config.pretrain_model.train.ngpu > 0:
        effective_batch_size = int(config.pretrain_model.train.batch_size / torch.cuda.device_count())

    loaders = [
        DataLoader(
            dataset=ds,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=config.pretrain_model.train.workers,
            #pin_memory=True,
            collate_fn=collater, #this modifies the data associated with sequences
        )
        for ds in dsets
    ]

    return loaders

