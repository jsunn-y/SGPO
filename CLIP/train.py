import os
import random
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5EncoderModel
from torch.utils.data import DataLoader
import esm

from model import AEModel
from dataset import CREEPDatasetMineBatch
from tokenization import SmilesTokenizer
import sys

class Logger:
    def __init__(self, filename, mode='a'):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def flatten_config(config, parent_key='', sep='_'):
    """Flatten nested config dictionary for easier access"""
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Config:
    """Config class to provide dot notation access to config values"""
    def __init__(self, config_dict):
        # Flatten the config for easier access
        flat_config = flatten_config(config_dict)
        
        # Set attributes for backward compatibility
        for key, value in flat_config.items():
            setattr(self, key, value)
        
        # Keep original nested structure
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

def do_CL(X, Y, args):
    if args.contrastive_learning_normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.contrastive_learning_CL_loss == 'EBM_NCE':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.contrastive_learning_CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.contrastive_learning_CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.contrastive_learning_T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.contrastive_learning_T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = (loss_pos + args.contrastive_learning_CL_neg_samples * loss_neg) / (1 + args.contrastive_learning_CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.contrastive_learning_CL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.contrastive_learning_T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc

def save_model(args, save_best=False):
    if args.output_output_model_dir is None:
        return
    
    if save_best:
        global optimal_loss
        print("save model with loss: {:.5f}".format(optimal_loss))
        model_file = "model.pth"
    else:
        model_file = "model_final.pth"

    saved_file_path = os.path.join(args.output_output_model_dir, "text_{}".format(model_file))
    torch.save(text_model.state_dict(), saved_file_path)
    
    saved_file_path = os.path.join(args.output_output_model_dir, "protein_{}".format(model_file))
    torch.save(protein_model.state_dict(), saved_file_path)

    saved_file_path = os.path.join(args.output_output_model_dir, "reaction_{}".format(model_file))
    torch.save(reaction_model.state_dict(), saved_file_path)
    
    saved_file_path = os.path.join(args.output_output_model_dir, "text2latent_{}".format(model_file))
    torch.save(text2latent_model.state_dict(), saved_file_path)
    
    saved_file_path = os.path.join(args.output_output_model_dir, "protein2latent_{}".format(model_file))
    torch.save(protein2latent_model.state_dict(), saved_file_path)

    saved_file_path = os.path.join(args.output_output_model_dir, "reaction2latent_{}".format(model_file))
    torch.save(reaction2latent_model.state_dict(), saved_file_path)

    saved_file_path = os.path.join(args.output_output_model_dir, "reaction2protein_facilitator_{}".format(model_file))
    torch.save(reaction2protein_facilitator_model.state_dict(), saved_file_path)

    saved_file_path = os.path.join(args.output_output_model_dir, "protein2reaction_facilitator_{}".format(model_file))
    torch.save(protein2reaction_facilitator_model.state_dict(), saved_file_path)

    return

def train(dataloader, protein_model, reaction_model, args):
    scaler = torch.cuda.amp.GradScaler()

    if args.training_verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc, accum_contrastive_loss, accum_generative_loss = 0, 0, 0, 0
    for batch_idx, batch in enumerate(L):
        protein_sequence_input_ids = torch.squeeze(batch["protein_sequence_input_ids"].to(device))
        protein_sequence_attention_mask = torch.squeeze(batch["protein_sequence_attention_mask"].to(device))
        reaction_sequence_input_ids = torch.squeeze(batch["reaction_sequence_input_ids"].to(device))
        reaction_sequence_attention_mask = torch.squeeze(batch["reaction_sequence_attention_mask"].to(device))
        
        with torch.cuda.amp.autocast():
            protein_embedding, protein_latent, protein_reconstruction = protein_aemodel(protein_sequence_input_ids, protein_sequence_attention_mask)
            reaction_embedding, reaction_latent, reaction_reconstruction = reaction_aemodel(reaction_sequence_input_ids, reaction_sequence_attention_mask)


            # manually train with just two modalities
            # calculate contrastive loss
            loss_01, acc_01 = do_CL(protein_latent, reaction_latent, args)
            loss_02, acc_02 = do_CL(reaction_latent, protein_latent, args)

            contrastive_loss = (loss_01 + loss_02) / 2
            contrastive_acc = (acc_01 + acc_02) / 2

            # calculate reconstruction loss
            criterion = nn.MSELoss()
            protein_recon_loss = criterion(protein_reconstruction, protein_embedding)
            reaction_recon_loss = criterion(reaction_reconstruction, protein_embedding)
            recon_loss = (protein_recon_loss + reaction_recon_loss) / 2

            #Used in the original ProteinDT paper to improve regularization
            # criterion = nn.MSELoss()
            # generative_loss = criterion(reaction2protein_repr, protein_repr) + criterion(protein2reaction_repr, reaction_repr)
            
            #loss = args.training_setup_alpha_contrastive * contrastive_loss + args.training_setup_alpha_generative * generative_loss
            loss = args.training_setup_alpha_contrastive * contrastive_loss + args.training_setup_alpha_recon * recon_loss
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accum_contrastive_loss += contrastive_loss.item()
        accum_generative_loss += generative_loss.item()
        accum_loss += loss.item()
        accum_acc += contrastive_acc
        if args.training_verbose and batch_idx % 100 == 0:
            print(loss.item(), contrastive_acc)
        
    accum_loss /= len(L)
    accum_contrastive_loss /= len(L)
    accum_generative_loss /= len(L)
    accum_acc /= len(L)
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(args, save_best=True)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tGenerative Loss: {:.5f}\tTotal Loss: {:.5f}\tTime: {:.5f}".format(
        accum_contrastive_loss, accum_acc, accum_generative_loss, accum_loss, time.time() - start_time))
    return

if __name__ == "__main__":
    # Parse command line arguments for config file path
    parser = argparse.ArgumentParser(description='Training script with YAML config')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to YAML config file (default: config.yaml)')
    parser.add_argument('--override', nargs='*', default=[],
                       help='Override config values (e.g., --override training.epochs=50 model.batch_size=32)')
    cmd_args = parser.parse_args()
    
    # Load configuration
    config = load_config(cmd_args.config)
    
    # Handle config overrides from command line
    for override in cmd_args.override:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use key=value format.")
        key, value = override.split('=', 1)
        
        # Convert value to appropriate type
        try:
            # Try to convert to number
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Try to convert to boolean
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            # Otherwise keep as string
        
        # Set nested config value
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    # Convert to Config object
    args = Config(config)
    
    # Set up logging
    if args.output_output_model_dir:
        log_file = "{}/log.txt".format(args.output_output_model_dir)
        # Make log file if it doesn't exist
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        sys.stdout = Logger(log_file, 'w')

    print("Configuration:", config)

    # Set random seeds
    random.seed(args.training_seed)
    os.environ['PYTHONHASHSEED'] = str(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    torch.cuda.manual_seed(args.training_seed)
    torch.cuda.manual_seed_all(args.training_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Load models using huggingface
    if args.model_protein_backbone_model == "ProtT5":
        protein_tokenizer = T5Tokenizer.from_pretrained(
            'Rostlab/prot_t5_xl_half_uniref50-enc', 
            do_lower_case=False, 
            cache_dir="data/pretrained_ProtT5"
        )
        protein_model = T5EncoderModel.from_pretrained(
            "Rostlab/prot_t5_xl_half_uniref50-enc", 
            cache_dir="data/pretrained_ProtT5"
        )
        protein_dim = 1024
    elif args.model_protein_backbone_model == "ESM2":
        #protein_model, protein_tokenizer = esm.pretrained.load_model_and_alphabet_local(args.esm_dir)
        protein_model, protein_tokenizer = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        protein_dim = 1280  # for 750M parameters
    
    # Load text model
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_text_backbone_model)
    text_model = AutoModel.from_pretrained(args.model_text_backbone_model)
    text_dim = text_model.config.hidden_size
    
    # Load reaction embeddings
    reaction_tokenizer = SmilesTokenizer.from_pretrained("../../CREEP/data/pretrained_rxnfp/vocab.txt")
    reaction_model = BertModel.from_pretrained("../../CREEP/data/pretrained_rxnfp")
    reaction_dim = 256

    # Create models (you'll need to define these based on your model.py)
    protein_aemodel = AEModel(protein_model, input_dim=protein_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    reaction_aemodel = AEModel(reaction_model, input_dim=reaction_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)

    device = torch.device("cuda:" + str(args.training_device)) if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)  # Uncomment when model is defined

    # Set up optimizer (you'll need to define model components based on your model.py)
    model_param_group = [
        {"params": protein_model.parameters(), "lr": args.learning_rates_protein_lr * args.learning_rates_protein_lr_scale},
        {"params": text_model.parameters(), "lr": args.learning_rates_text_lr * args.learning_rates_text_lr_scale},
        {"params": reaction_model.parameters(), "lr": args.learning_rates_reaction_lr * args.learning_rates_reaction_lr_scale},
        # Add other model components...
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.learning_rates_decay)
    optimal_loss = 1e10

    # Create dataset
    dataset = CREEPDatasetMineBatch(
        dataset_path=args.data_dataset_path,
        train_file="data/splits/" + args.data_train_split + ".csv",
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.sequence_lengths_protein_max_sequence_len,
        text_max_sequence_len=args.sequence_lengths_text_max_sequence_len,
        reaction_tokenizer=reaction_tokenizer,
        reaction_max_sequence_len=args.sequence_lengths_reaction_max_sequence_len,
        n_neg=args.training_batch_size - 1
    )
    
    # Batch size is 1 when mining the batch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.training_num_workers)

    # Training loop
    for e in range(1, args.training_epochs + 1):
        print("Epoch {}".format(e))
        train(dataloader, args)