import os
from omegaconf import OmegaConf
import pickle
import pandas as pd
import hydra
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
import wandb
from models.pretraining.collaters import collate_fn_mapping
from util.seed import set_seed  
import tqdm
from dataset.protein import ProteinPredictorDataset
import copy

def sample(config, n_samples, algorithm, dataset, batch_size, round):
    #calculate number of batches
    max_iterations = n_samples // batch_size
    sample_steps = range(0, max_iterations)
    
    # for i in batch_steps:
    for step in tqdm.tqdm(sample_steps):
        _, detokenized = algorithm.inference(num_samples=batch_size, detokenize=True)
        n_new_samples = dataset.update_data(detokenized, n_samples, BO=config.BO, round=round,unique_only=config.unique_only)
    return dataset

@hydra.main(version_base="1.3", config_path="configs", config_name="pareto_config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.batch_size

    summary_df = pd.DataFrame()
    exp_dir = os.path.join(config.problem.exp_dir, config.data.name, config.pretrained_ckpt.split('/')[0], config.exp_name, config.algorithm.name)
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    summary_save_path = os.path.join(exp_dir, 'pareto_hyperparameter_summary.csv')
    data_config = config.data
    seq_len = data_config.seq_len

    sample_batch_size = config.num_samples if config.problem.sample_batch_size > config.num_samples else config.problem.sample_batch_size

    for i in range(config.n_repeats):
        save_dir = os.path.join(exp_dir, "models", f"repeat{i}")
        ### Sample the initial random data to be used for everything ###
        for task, n_max_mutations in zip(["unconstrained"], [None]):
            set_seed(config.seed+i)
            save_dir = os.path.join(config.problem.exp_dir, config.data.name, "initial_sample_d3pm", task)
            os.makedirs(save_dir, exist_ok=True)
        
            fasta_save_path = os.path.join(save_dir, f'generated_{i}.fasta')

            #run sampling for the initial round
            if not os.path.exists(fasta_save_path):
                net = instantiate(OmegaConf.load("configs/model/d3pm.yaml").model,model_name=f"d3pm_finetune/{config.data.name}", seq_len=seq_len, device=device)
                algorithm = instantiate(OmegaConf.load("configs/algorithm/cls_guidance_discrete.yaml").method, data_config=config.data, temperature=0., n_max_mutations=n_max_mutations, net=net, forward_op=None) #forward_op not required if guidance set to zero
                
                #TODO: remove non unique sequences from this initial set?
                dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer)
                sample(config, config.num_steer, algorithm, dataset, batch_size, 0)
                
                #save dataset.seqs to fasta
                with open(fasta_save_path, 'w') as f:
                    for j, seq in enumerate(dataset.seqs):
                        f.write(f">{j}\n")
                        f.write(f"{seq}\n")
                print(f"Saved generated sequences to {fasta_save_path}")

            data_config = config.data
            seq_len = data_config.seq_len

            if 'NOS' in config.algorithm.name:
                recursive = False if "mdlm" in config.model.name or "udlm" in config.model.name else True
                net = instantiate(config.model.model, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device, _recursive_=recursive) 
                #time_conditioned = True
                time_conditioned = config.algorithm.time_conditioned if 'time_conditioned' in config.algorithm else True

                if 'NOS' in config.algorithm.name and 'continuous' in config.model.name:
                    #freeze the transformer weights
                    for param in net.model.parameters():
                        param.requires_grad = False

                    classifier = instantiate(config.problem.model, data_config=data_config, _recursive_=recursive)
                    algorithm = instantiate(config.algorithm.method, nos_stability_coef=None, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config)
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                elif 'NOS' in config.algorithm.name:
                    pretrained_backbone = copy.deepcopy(net.model.backbone)
                    # Remove the last layer for the classifier
                    if hasattr(pretrained_backbone, 'output_layer'):  #DiT
                        delattr(pretrained_backbone, 'output_layer')
                    #freeze backbone for NOS
                    for param in pretrained_backbone.parameters():
                        param.requires_grad = False

                    classifier = instantiate(config.problem.model, tokenizer=net.tokenizer, pretrained_backbone=pretrained_backbone, _recursive_=recursive)
                    algorithm = instantiate(config.algorithm.method, nos_stability_coef=None, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config) 
                    # algorithm = instantiate(config.algorithm.method, nos_stability_coef=0., n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config) #dummy to get the project_fn, replace later
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                else:
                    raise ValueError(f"Unknown algorithm {config.algorithm.name}.")

            #initialize data once for all guidance parameters
            print(f'Initializing data based on presampled sequences from fasta.')
            dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer, from_fasta=fasta_save_path)
            dataloader = DataLoader(dataset, batch_size=config.problem.train.batch_size, collate_fn=collate_fn, shuffle=True)
            dataset.summary_df["nos_stability_coeff"] = 0.
            dataset.summary_df["nos_num_steps"] = 0
            dataset.summary_df["nos_step_size"] = 0.
            dataset.summary_df["repeat"] = i
            dataset.summary_df["task"] = task
            summary_df = pd.concat([summary_df, dataset.summary_df], axis=0)

            #train model once for guidance-based methods
            if 'NOS' in config.algorithm.name:
                print(f"Training predictor model.")
                #Classifier guidance or posterior sampling
                classifier = instantiate(config.problem.train_function, classifier, net, dataloader, config.problem.train, save_dir, project_fn=algorithm.project, time_conditioned=time_conditioned)

            nos_steps = config.algorithm.nos_steps if 'nos_steps' in config.algorithm else [5, 10]
            nos_step_sizes = config.algorithm.nos_step_sizes if 'nos_step_sizes' in config.algorithm else [0.5, 2., 5.]
            nos_stability_coefs = config.algorithm.nos_stability_coefs if 'nos_stability_coefs' in config.algorithm else [0.1, 1., 10., 100., 1000.]
            
            for nos_num_steps in nos_steps:
                for nos_step_size in nos_step_sizes:
                    for nos_stability_coef in nos_stability_coefs:
                       
                        set_seed(config.seed + i)

                        print(f'Initializing data based on presampled sequences from fasta.')
                        dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer, from_fasta=fasta_save_path)
                        dataloader = DataLoader(dataset, batch_size=config.problem.train.batch_size, collate_fn=collate_fn, shuffle=True)
                        
                        if 'NOS' in config.algorithm.name:
                            #algorithm = instantiate(config.algorithm.method, nos_step_size=guidance_param, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config)
                            algorithm = instantiate(config.algorithm.method, nos_stability_coef=nos_stability_coef, num_nos_steps=nos_num_steps, nos_step_size=nos_step_size, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config)
                            algorithm.update_model(classifier)
                        else:
                            ValueError(f"Unknown algorithm {config.algorithm.name}.")
                        
                        #Sample new sequences
                        dataset = sample(config, config.num_samples, algorithm, dataset, sample_batch_size, 1)
                        dataset.summary_df["repeat"] = i
                        dataset.summary_df["nos_stability_coeff"] = nos_stability_coef
                        dataset.summary_df["nos_num_steps"] = nos_num_steps
                        dataset.summary_df["nos_step_size"] = nos_step_size
                        dataset.summary_df["task"] = task

                        summary_df = pd.concat([summary_df, dataset.summary_df[dataset.summary_df["round"] == 1]], axis=0)
                        summary_df.to_csv(summary_save_path, index=False)


if __name__ == "__main__":
    main()
