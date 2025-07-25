import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, Encoding
from typing import Any, Dict, List, Set, Tuple, Union

from models.pretraining.model.progen2.model import ProGenForCausalLM
from models.pretraining.model.progen2.tokenizer import get_tokenizer
from models.pretraining.collaters import collate_fn_mapping
from dataset.protein import ProteinPredictorDataset

device = 'cuda'
# os.chdir('../')
<<<<<<< HEAD
proteins =  ["TrpB", "CreiLOV"] #["GB1"] #use the original ProGen2 model or the finetuned version
#priors = ["random", "target", "continuous", "d3pm_finetune", "mdlm", "causalLM_finetune"] 
priors = ["random", "target", "continuous", "continuous_ESM", "d3pm", "d3pm_finetune", "udlm", "mdlm","causalLM_finetune"] # "mdlm_long", "mdlm_short",
=======
proteins = ["GB1"] #["TrpB", "CreiLOV"] #use the original ProGen2 model or the finetuned version
priors = ["random", "target", "continuous", "d3pm_finetune", "mdlm", "causalLM_finetune"] 
# priors = ["random", "continuous", "continuous_ESM_head", "d3pm_finetune", "udlm", "mdlm","causalLM_finetune"] # "mdlm_long", "mdlm_short",
>>>>>>> 032bd50 (bug fixes and running new analysis for GB1)
# problems = ["random", "protein_classifier_continuous", "protein_classifier", "protein_classifier", "protein_DPO"]
tasks = ["unconstrained"] #constrained
types = [False] #True #whether to use the finetuned progen model or not
sample_batch_size = 50
tqdm_iter = tqdm(range(len(priors) * len(tasks) * len(proteins) * len(types)), desc="Calculating perplexity")

prefix = "exps/protein"

for protein in proteins:
    df = pd.DataFrame(columns=['sequence', 'perplexity', 'prior', 'task', 'finetuned'])

    for use_finetuned in types: 
        if use_finetuned:
            ckpt_file = f'{prefix}/checkpoints/causalLM_finetune/{protein}/best'
            model = ProGenForCausalLM.from_pretrained(ckpt_file)
            tokenizer = get_tokenizer()
        else:
            #download the pretrained model from the huggingface model hub and cache it locally
            model = ProGenForCausalLM.from_pretrained("jsunn-y/ProCALM", subfolder="{}-{}".format("progen2", "base"))
            tokenizer = Tokenizer.from_pretrained("jsunn-y/ProCALM")

        model.to(device)
        model.eval()

        collate_fn = collate_fn_mapping['DPO'](tokenizer=tokenizer)

        data_config = OmegaConf.load(f"configs/data/{protein}.yaml")

        for prior in priors:
            # problem_config = OmegaConf.load(f"configs/problem/{problem}.yaml")

            for task in tasks:
                if prior == "random":
                    fasta_path = os.path.join(prefix, "exps", protein, "baseline", task, "generated.fasta")
                elif prior == "target":
                    fasta_path = os.path.join("data", f"{protein}/MSA_sample.fasta")
                elif prior == "target_LM":
                    #currently not supported
                    fasta_path = os.path.join("data", f"{protein}/MSA_aligned_sample.fasta")
                else:
                    # fasta_path = os.path.join("exps", "protein", protein, prior, "uncond", "generated.fasta")
                    fasta_path = os.path.join(prefix, "exps", protein, prior, "prior_sample", task, "generated.fasta")

                #use the same dataset as those used during guidance
                #probably has some redundant y-value intialization but that's fast
                dataset = ProteinPredictorDataset(data_config, tokenizer=tokenizer, from_fasta=fasta_path)

                #TODO: if you don't want 1 and 2 at thebeginning, then change the collate_fn to not include them
                loader = DataLoader(dataset, batch_size=sample_batch_size, collate_fn=collate_fn, shuffle=False) #make sure shuffle is on false here

                all_sequences = []
                all_perplexities = []

                with torch.no_grad():
                    for batch in loader:
                        sequences = batch['sequences'] 
                        include = {"input_ids", "attention_mask", "labels"}
                        processed_batch = {k: v for k, v in batch.items() if k in include}

                        for key in {"input_ids", "attention_mask", "labels"}:
                            processed_batch[key] = processed_batch[key].to(device)
                        
                        batch_size = processed_batch["input_ids"].shape[0]

                        outputs = model.forward(**processed_batch)
                        losses = outputs.all_losses
                        all_sequences.extend(sequences)

                        losses = losses.reshape(batch_size, -1).cpu().numpy()
                        #take the mean along each row, but ignore zeros (these are the pad tokens)
                        losses = list(np.exp(losses.sum(axis=1) / (losses != 0).sum(axis=1)))
                        all_perplexities.extend(losses)
                        
                update_df = pd.DataFrame({'sequence': all_sequences, 'perplexity': all_perplexities, 'prior': prior, 'task': task, 'finetuned': use_finetuned, 'fitness': dataset.y.tolist()})
                df = pd.concat([df, update_df])
                folder = f'exps/protein/perplexity/{protein}'
                os.makedirs(folder, exist_ok=True)
                df.to_csv(f'{folder}/perplexity.csv', index=False)
                tqdm_iter.update(1)
#
