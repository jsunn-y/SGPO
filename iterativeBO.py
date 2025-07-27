import os
import importlib
from omegaconf import OmegaConf
import pickle
import hydra
from hydra.utils import instantiate
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import wandb
from models.pretraining.collaters import collate_fn_mapping
from util.seed import set_seed  
import tqdm
import copy
from dataset.protein import ProteinPredictorDataset
from training.train_GP_classifier import train_classifier, get_thompson_sample


def sample(config, algorithm, dataset, batch_size, save_path, round):
    #batch_steps = range(0, 2*config.num_samples, batch_size)
    max_iterations = max(5 * config.num_samples // batch_size, 5)
    sample_steps = range(0, config.num_samples)
    samples_pbar = tqdm.tqdm(sample_steps)
    
    n_total_new_samples = 0
    iterations = 0
    # for i in batch_steps:
    with samples_pbar as pbar:
        pbar.set_description(f"Sampling {config.num_samples} samples")
        while n_total_new_samples < config.num_samples and iterations < max_iterations: #don't sample more than twice the number of target samples
            #TODO check if there is something weird happening here
            n_needed = config.num_samples - n_total_new_samples
            _, detokenized = algorithm.inference(num_samples=batch_size, detokenize=True)
            n_new_samples = dataset.update_data(detokenized, n_needed, round=round, BO=config.BO)
            n_total_new_samples += n_new_samples
            iterations += 1
            pbar.update(n_new_samples)
    
    if n_total_new_samples < config.num_samples:
        print(f"Only generated {n_total_new_samples} samples, less than the desired {config.num_samples}.")
    
    #TODO: add some random samples if the generator doesn't have enough unique ones
    
    # dataset.save_data(save_path)
    return

@hydra.main(version_base="1.3", config_path="configs", config_name="iterativeBO_config") #iterativeBO_noUQ_config
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    summary_df = pd.DataFrame()
    name = config.algorithm.name + "_GP" if "GP" in config.problem.name else config.algorithm.name
    exp_dir = os.path.join(config.problem.exp_dir, config.data.name, config.pretrained_ckpt.split('/')[0], config.exp_name, name)
    os.makedirs(exp_dir, exist_ok=True)
    # save config 
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))
    summary_save_path = os.path.join(exp_dir, 'BO_summary.csv')
    data_config = config.data
    seq_len = data_config.seq_len

    sample_batch_size = config.num_samples if config.problem.sample_batch_size > config.num_samples else config.problem.sample_batch_size

    recursive = False if "mdlm" in config.model.name or "udlm" in config.model.name else True
    for i in range(config.n_repeats):
        ### Sample the initial random data to be used for everything ###
        for task, n_max_mutations in zip(["unconstrained"], [None]): #constrained #4
            set_seed(config.seed+i)

            data_config = config.data
            seq_len = data_config.seq_len
            time_conditioned = config.algorithm.time_conditioned if 'time_conditioned' in config.algorithm else True #train classifier on noisy data for most methods, except DAPS
                
            if 'cls_guidance' in config.algorithm.name or 'DAPS' in config.algorithm.name or 'NOS' in config.algorithm.name:
                net = instantiate(config.model.model, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device, _recursive_=recursive) #load pretrained model

                if 'cls_guidance' in config.algorithm.name:
                    #classifier = instantiate(config.problem.model, data_config=data_config, device=device)
                    algorithm = instantiate(config.algorithm.method, temperature=0., n_max_mutations=n_max_mutations, net=net, forward_op=None, data_config=data_config) #dummy to get the project_fn, replace later
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                elif 'DAPS' in config.algorithm.name:
                    #classifier = instantiate(config.problem.model, data_config=data_config, device=device)
                    algorithm = instantiate(config.algorithm.method, alpha=0., n_max_mutations=n_max_mutations, net=net, forward_op=None, data_config=data_config) #dummy to get the project_fn, replace later
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                elif 'NOS' in config.algorithm.name and 'continuous' in config.model.name:
                    #freeze the transformer weights
                    for param in net.model.parameters():
                        param.requires_grad = False

                    #classifier = instantiate(config.problem.model, data_config=data_config, _recursive_=recursive)
                    algorithm = instantiate(config.algorithm.method, nos_stability_coef=None, n_max_mutations=n_max_mutations, net=net, forward_op=None, data_config=data_config)
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                elif 'NOS' in config.algorithm.name:
                    pretrained_backbone = copy.deepcopy(net.model.backbone)
                    # Remove the last layer for the classifier
                    if hasattr(pretrained_backbone, 'output_layer'):  #DiT
                        delattr(pretrained_backbone, 'output_layer')
                    #freeze backbone for NOS
                    for param in pretrained_backbone.parameters():
                        param.requires_grad = False

                    #classifier = instantiate(config.problem.model, tokenizer=net.tokenizer, pretrained_backbone=pretrained_backbone, _recursive_=recursive)
                    algorithm = instantiate(config.algorithm.method, nos_stability_coef=None, n_max_mutations=n_max_mutations, net=net, forward_op=None, data_config=data_config) 
                    # algorithm = instantiate(config.algorithm.method, nos_stability_coef=0., n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config) #dummy to get the project_fn, replace later
                    collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)

            elif 'DPO' in config.algorithm.name:
                net = instantiate(config.model.model, load_ref_model=True, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device) 
                #initiate algo once, doesn't need to be changed for DPO
                algorithm = instantiate(config.algorithm.method, net=net, data_config=data_config, n_max_mutations=n_max_mutations)
                collate_fn = collate_fn_mapping['DPO'](tokenizer=net.tokenizer)

            save_path = os.path.join(exp_dir, 'BO_summary.csv')

            #initialize data from prior without guidance
            print(f'Sampling unconditionally for initial round')
            if config.random_init:
                dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer, n_random_init=config.num_samples)
                # dataset.save_data(save_path)
            else:
                dataset = instantiate(config.problem.data, data_config=data_config, tokenizer=net.tokenizer)
                sample(config, algorithm, dataset, sample_batch_size, save_path, round=0)

            dataloader = DataLoader(dataset, batch_size=config.problem.train.batch_size, collate_fn=collate_fn, shuffle=True)
            dataset.summary_df["guidance_param"] = 0.
            dataset.summary_df["repeat"] = i
            dataset.summary_df["task"] = task
            summary_df = pd.concat([summary_df, dataset.summary_df], axis=0)

            #### Now get ready to sample unconditionally using the ideal guidance parameter found in pareto.py ####
            if 'cls_guidance' in config.algorithm.name or 'DAPS' in config.algorithm.name or 'NOS' in config.algorithm.name:
                #TODO: check to make sure the algorithm is getting updated correctly
                algorithm = instantiate(config.algorithm.method, n_max_mutations=n_max_mutations, net=net, forward_op=None, data_config=data_config) #dummy to get the project_fn, replace later
                collate_fn = collate_fn_mapping['cls_guidance'](tokenizer=net.tokenizer)
                #not sure if this is necessary, only for saving results
                if 'cls_guidance' in config.algorithm.name:
                    guidance_param = config.algorithm.method.temperature
                elif 'DAPS' in config.algorithm.name:
                    guidance_param = config.algorithm.method.alpha
                elif 'NOS' in config.algorithm.name:
                    guidance_param = config.algorithm.method.nos_stability_coef
                
            for round in range(config.num_rounds):
                #TODO implement better model tracking in wandb and save the models?
                #TODO check if the small batches are still computationally efficient
                #train model once for guidance-based methods
                if "DPO" in config.algorithm.name:
                    sample_batch_size = config.num_samples if config.problem.sample_batch_size > config.num_samples else config.problem.sample_batch_size
                else:
                    if "GP" in config.problem.name:
                        config.n_ensemble = None #override so ensemble is not used
                        sample_batch_size = 10 #a few at a time for GP, could do 5 or 2 but may not use the whole gpu
                    else:
                        sample_batch_size = int(config.num_samples/config.n_ensemble) 

                    if sample_batch_size > config.problem.sample_batch_size:
                        sample_batch_size = int(sample_batch_size / 2)
                print(f"Sample batch size: {sample_batch_size}")

                max_iterations = max(5 * config.num_samples // sample_batch_size, 5) #oversample up to a certain amount
                sample_steps = range(0, config.num_samples)
                
                n_total_new_samples = 0
                iterations = 0
                # for i in batch_steps:

                #train ensemble of predictor models/finetune models
                if 'cls_guidance' in config.algorithm.name or 'DAPS' in config.algorithm.name or 'NOS' in config.algorithm.name:
                    print(f"Training predictor model.")
                    #classifiers = []
                    processes = []
                    mp.set_start_method("spawn", force=True)
                    save_dir = os.path.join(exp_dir, "models", f"repeat{i}", f"round{round}")
                    os.makedirs(save_dir, exist_ok=True)

                    if config.n_ensemble is None:
                        #train a GP
                        classifier_GP = train_classifier(model=net, dataloader=dataloader,  data_config=data_config, model_config=config.problem.model, train_config=config.problem.train, save_dir=save_dir, project_fn=None, time_conditioned=time_conditioned)
                        # classifier_GP = hydra.utils.instantiate(config.problem.train_function, model=net, embedder=embedder, dataloader=dataloader,  model_config=config.problem.model, train_config=config.problem.train, project_fn=None, time_conditioned=time_conditioned)
                    else:
                        for ensemble_idx in range(config.n_ensemble):
                            set_seed(config.seed + round*config.n_ensemble + ensemble_idx)

                            if 'NOS' in config.algorithm.name and 'continuous' in config.model.name:

                                classifier = instantiate(config.problem.model, data_config=data_config, _recursive_=recursive)

                                #algorithm = instantiate(config.algorithm.method, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config)
                            elif 'NOS' in config.algorithm.name:

                                classifier = instantiate(config.problem.model, tokenizer=net.tokenizer, pretrained_backbone=pretrained_backbone, _recursive_=recursive)
                            else:
                                #CG and DAPS
                                classifier = instantiate(config.problem.model, data_config=data_config, device=device)
                            
                            #Classifier guidance or posterior sampling
                            # classifier = instantiate(config.problem.train_function, classifier, net, dataloader, train_config=config.problem.train, project_fn=algorithm.project)
                            # classifiers.append(classifier.to("cpu"))
                            
                            #train models in parallel
                            module_name, func_name = config.problem.train_function._target_.rsplit(".", 1)
                            train_function = getattr(importlib.import_module(module_name), func_name)
                            p = instantiate(config.process, target=train_function, args=(classifier, net, dataloader, config.problem.train, save_dir, algorithm.project, ensemble_idx, time_conditioned))
                            p.start()
                            processes.append(p)
                        
                        for p in processes:
                            p.join()
                        
                        #load the saved models
                        classifiers = []
                        for ensemble_idx in range(config.n_ensemble):
                            classifier = torch.load(os.path.join(save_dir, f"classifier_{ensemble_idx}.pt"), weights_only=False)
                            classifiers.append(classifier)

                elif 'DPO' in config.algorithm.name:
                    #nets = []
                    for ensemble_idx in range(1):
                        #TODO: figure out how to introduce randomness here? Since NN initialization is deterministic in this setting
                        #for now, just run with a single model
                        set_seed(config.seed + round*config.n_ensemble + ensemble_idx)
                        net = instantiate(config.model.model, load_ref_model=True, model_name=config.pretrained_ckpt, seq_len=seq_len, device=device) 
                        print(f"Finetuning model with DPO.")
                        #Reinforcement learning (finetuning)

                        #for now, manually overide ideal beta parameter for DPO
                        if data_config.name == "CreiLOV":
                            config.problem.train.beta = 0.25 

                        net = instantiate(config.problem.train_function, net=net, train_loader=dataloader, train_config=config.problem.train)
                        #nets.append(net.to("cpu"))
                    guidance_param = config.problem.train.beta
                else:
                    raise NotImplementedError(f"Unknown algorithm {config.algorithm.name}.")

                #generate samples with "thompson sampling" and steering/guidance
                set_seed(config.seed + round)
                samples_pbar = tqdm.tqdm(sample_steps)
                with samples_pbar as pbar:
                    pbar.set_description(f"Sampling {config.num_samples} samples")
                    while n_total_new_samples < config.num_samples and iterations < max_iterations:
                        if 'cls_guidance' in config.algorithm.name or 'DAPS' in config.algorithm.name or 'NOS' in config.algorithm.name:
                            #if thompson sampling, random sample one model
                            if config.thompson_sampling:
                                #GP
                                if config.n_ensemble is None:
                                    classifier = get_thompson_sample(classifier_GP, data_config, config.problem.model, time_conditioned=time_conditioned, device=device)
                                #Frequentist ensemble
                                else:
                                    classifier = classifiers[np.random.randint(0, config.n_ensemble)].to(device)
                            else:
                                raise NotImplementedError("Only Thompson sampling is supported for now.")

                            algorithm = instantiate(config.algorithm.method, n_max_mutations=n_max_mutations, net=net, forward_op=classifier, data_config=data_config)
                            algorithm.update_model(classifier)
                    
                        #train model for each parameter for finetuning-based methods
                        elif 'DPO' in config.algorithm.name:
                            #if thompson sampling, random sample one model
                            if config.thompson_sampling:
                                #net = nets[np.random.randint(0, config.n_ensemble)].to(device)
                                pass
                            else:
                                raise NotImplementedError("Only Thompson sampling is supported for now.")
                            algorithm.update_model(net)
                        else:
                            raise NotImplementedError(f"Unknown algorithm {config.algorithm.name}.")
                            
                        #Sample new sequences
                        print(f'Sampling for steered round with guidance parameter {guidance_param}.')
                        n_needed = config.num_samples - n_total_new_samples
                        _, detokenized = algorithm.inference(num_samples=sample_batch_size, detokenize=True)
                        n_new_samples = dataset.update_data(detokenized, n_needed, round=round+1, BO=config.BO)
                        n_total_new_samples += n_new_samples
                        iterations += 1
                        pbar.update(n_new_samples)
                
                if n_total_new_samples < config.num_samples:
                    print(f"Only generated {n_total_new_samples} samples, less than the desired {config.num_samples}.")

                new_dataset = dataset.summary_df[dataset.summary_df["round"] == round+1]
                new_dataset["repeat"] = i
                new_dataset["guidance_param"] = guidance_param
                new_dataset["task"] = task

                summary_df = pd.concat([summary_df, new_dataset], axis=0)
                summary_df.to_csv(summary_save_path, index=False)


if __name__ == "__main__":
    main()