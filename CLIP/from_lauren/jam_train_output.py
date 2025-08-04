import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
import argparse
import csv

from data_loader import *
from models import *
from loss_objective import *
from jam_eval import *

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def train_model_spread(train_loader, val_loader, test_loader, max_alpha=0.5, latent_dim=256, hidden_dims=(512, 512), epochs=200):
    img_batch, pos_txt_batch, _ = next(iter(train_loader))
    input_dim_img  = img_batch.size(1)
    input_dim_text = pos_txt_batch.size(1)

    img_ae  = AE(in_dim=input_dim_img, latent=latent_dim, hidden_dims=hidden_dims, dropout=0.1, tie_weights=False).to(device)
    txt_ae  = AE(in_dim=input_dim_text, latent=latent_dim, hidden_dims=hidden_dims, dropout=0.1, tie_weights=False).to(device)
    
    align_loss = Lspread(alpha=0.0).to(device)
    recon_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(list(img_ae.parameters()) + list(txt_ae.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs*2, eta_min=1e-7)
    
    wandb.watch(img_ae, log="all", log_freq=100)
    wandb.watch(txt_ae, log="all", log_freq=100)
    
    def get_lambda_recon(epoch, total, start=1.0, end=0.1):
        t = epoch / total
        return start * (1 - t) + end * t
    
    epoch_records = []
    for epoch in range(epochs):
        alpha = max_alpha
        align_loss.alpha = alpha
        lambda_recon = get_lambda_recon(epoch, epochs)
        wandb.log({"alpha": alpha, "lambda": lambda_recon, "epoch": epoch})

        img_ae.train()
        txt_ae.train()

        train_loss = 0.0
        train_spread_loss = 0.0
        train_recon_loss = 0.0
        for img_embs, pos_txt_embs, neg_txt_embs in train_loader:
            img_embs = img_embs.to(device)
            pos_txt_embs = pos_txt_embs.to(device)
            neg_txt_embs = neg_txt_embs.to(device)

            latent_img, recon_img = img_ae(img_embs)
            latent_txt_pos, recon_txt = txt_ae(pos_txt_embs)
            latent_txt_neg, _ = txt_ae(neg_txt_embs)

            lrecon_img = recon_loss(recon_img, img_embs)
            lrecon_text = recon_loss(recon_txt, pos_txt_embs)
            lrecon = (lrecon_img + lrecon_text) / 2
            lspread = align_loss(latent_img, latent_txt_pos, latent_txt_neg)
            loss = lspread + lambda_recon * lrecon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_ae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(txt_ae.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_spread_loss += lspread.item()
            train_recon_loss += lrecon.item()

        img_ae.eval()
        txt_ae.eval()

        val_loss = 0.0
        val_spread_loss = 0.0
        val_recon_loss  = 0.0

        with torch.no_grad():
            for img_embs, pos_txt_embs, neg_txt_embs in val_loader:
                img_embs = img_embs.to(device)
                pos_txt_embs = pos_txt_embs.to(device)
                neg_txt_embs = neg_txt_embs.to(device)

                latent_img, recon_img = img_ae(img_embs)
                latent_pos, recon_txt = txt_ae(pos_txt_embs)
                latent_neg, _ = txt_ae(neg_txt_embs)

                lrecon_img = recon_loss(recon_img,  img_embs)
                lrecon_text = recon_loss(recon_txt, pos_txt_embs)
                lrecon = (lrecon_img + lrecon_text) / 2
                lspread = align_loss(latent_img, latent_pos, latent_neg)
                loss = lspread + lambda_recon * lrecon

                val_loss += loss.item()
                val_spread_loss += lspread.item()
                val_recon_loss += lrecon.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_spread = train_spread_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_spread = val_spread_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})

        val_eval_interval = 5
        best_val_i2t   = -1.0
        epochs_no_improve = 0
        patience = 5
        if (epoch % val_eval_interval) == 0:
            val_i2t, val_t2i, easy, medium = evaluate_retrieval(img_ae, txt_ae, val_loader)
            wandb.log({"val_recall_i2t": val_i2t, "val_recall_t2i": val_t2i, "val_easy_recall": easy, "val_medium_recall": medium})
            if val_i2t > best_val_i2t + 1e-4:
                best_val_i2t = val_i2t
                epochs_no_improve = 0
                torch.save({
                    "img_ae": img_ae.state_dict(),
                    "txt_ae": txt_ae.state_dict(),
                    "epoch": epoch,
                    "val_i2t": val_i2t,
                }, "best_val_recall.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"val recall plateaued: early stopping @ epoch {epoch}")
                    break
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss {avg_train_loss:.4f} | "
              f"train_spread: {avg_train_spread:.4f} | "
              f"train_recon: {avg_train_recon:.4f} | "
              f"val_loss {avg_val_loss:.4f} | "
              f"val_spread: {avg_val_spread:.4f} | "
              f"val_recon: {avg_val_recon:.4f} | "
              f"lr: {current_lr:.2e}")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_spread_loss": avg_train_spread,
            "train_recon_loss": avg_train_recon,
            "val_loss": avg_val_loss,
            "val_spread_loss": avg_val_spread,
            "val_recon_loss": avg_val_recon,
            "learning_rate": current_lr,
        })
        
        img2txt_acc, txt2img_acc, easy, medium = evaluate_retrieval(img_ae, txt_ae, test_loader)
        record = dict(epoch=epoch, alpha=alpha, img2txt_acc=img2txt_acc, txt2img_acc=txt2img_acc, easy=easy, medium=medium)
        epoch_records.append(record)

    print("Training complete!")
    return epoch_records

def train_model_baseline(train_loader, val_loader, test_loader, latent_dim=256, hidden_dims=(512, 512), align_type="lcon", epochs=200):
    img_batch, pos_txt_batch, _ = next(iter(train_loader))
    input_dim_img  = img_batch.size(1)
    input_dim_text = pos_txt_batch.size(1)

    img_ae  = AE(in_dim=input_dim_img, latent=latent_dim, hidden_dims=hidden_dims, dropout=0.1, tie_weights=False).to(device)
    txt_ae  = AE(in_dim=input_dim_text, latent=latent_dim, hidden_dims=hidden_dims, dropout=0.1, tie_weights=False).to(device)
    
    if align_type.lower() == "lcon":
        align_loss = Lcon().to(device)
    elif align_type.lower() == "lconneg":
        align_loss = LconNEG().to(device)
    else:
        raise ValueError(f"Unknown align_type: {align_type}")
    
    recon_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(list(img_ae.parameters()) + list(txt_ae.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs*2, eta_min=1e-6)

    wandb.watch(img_ae, log="all", log_freq=100)
    wandb.watch(txt_ae, log="all", log_freq=100)
    
    def get_lambda_recon(epoch, total, start=1.0, end=0.1):
        t = epoch / total
        return start * (1 - t) + end * t
    
    epoch_records = []
    for epoch in range(epochs):
        lambda_recon = get_lambda_recon(epoch, epochs)
        wandb.log({"lambda": lambda_recon, "epoch": epoch})

        img_ae.train()
        txt_ae.train()

        train_loss = 0.0
        train_cont_loss = 0.0
        train_recon_loss = 0.0

        for img_embs, pos_txt_embs, neg_txt_embs in train_loader:
            img_embs = img_embs.to(device)
            pos_txt_embs = pos_txt_embs.to(device)
            neg_txt_embs = neg_txt_embs.to(device)

            latent_img, recon_img = img_ae(img_embs)
            latent_txt_pos, recon_txt = txt_ae(pos_txt_embs)
            latent_txt_neg, _ = txt_ae(neg_txt_embs)

            lrecon_img = recon_loss(recon_img, img_embs)
            lrecon_text = recon_loss(recon_txt, pos_txt_embs)
            lrecon = (lrecon_img + lrecon_text) / 2

            if align_type.lower() == "lcon":
                lcont = align_loss(latent_img, latent_txt_pos)
            else:
                lcont = align_loss(latent_img, latent_txt_pos, latent_txt_neg)
            loss = lcont + lambda_recon * lrecon
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_ae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(txt_ae.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cont_loss += lcont.item()
            train_recon_loss += lrecon.item()
        
        img_ae.eval()
        txt_ae.eval()
        
        val_loss = 0.0
        val_cont_loss = 0.0
        val_recon_loss  = 0.0
        
        with torch.no_grad():
            for img_embs, pos_txt_embs, neg_txt_embs in val_loader:
                img_embs = img_embs.to(device)
                pos_txt_embs = pos_txt_embs.to(device)
                neg_txt_embs = neg_txt_embs.to(device)
                
                latent_img, recon_img = img_ae(img_embs)
                latent_pos, recon_txt = txt_ae(pos_txt_embs)
                latent_neg, _ = txt_ae(neg_txt_embs)
                
                lrecon_img = recon_loss(recon_img,  img_embs)
                lrecon_text = recon_loss(recon_txt, pos_txt_embs)
                lrecon = (lrecon_img + lrecon_text) / 2
                if align_type.lower() == "lcon":
                    lcont = align_loss(latent_img, latent_pos)
                else:
                    lcont = align_loss(latent_img, latent_pos, latent_neg)
                loss = lcont + lambda_recon * lrecon
                
                val_loss += loss.item()
                val_cont_loss += lcont.item()
                val_recon_loss += lrecon.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cont = train_cont_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_cont = val_cont_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"learning_rate": current_lr})

        val_eval_interval = 5
        best_val_i2t   = -1.0
        epochs_no_improve = 0
        patience = 3
        if (epoch % val_eval_interval) == 0:
            val_i2t, val_t2i, easy, medium = evaluate_retrieval(img_ae, txt_ae, val_loader)
            wandb.log({"val_recall_i2t": val_i2t, "val_recall_t2i": val_t2i, "val_easy_recall": easy, "val_medium_recall": medium})
            if val_i2t > best_val_i2t + 1e-4:
                best_val_i2t = val_i2t
                epochs_no_improve = 0
                torch.save({
                    "img_ae": img_ae.state_dict(),
                    "txt_ae": txt_ae.state_dict(),
                    "epoch": epoch,
                    "val_i2t": val_i2t,
                }, "best_val_recall.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"val recall plateaued: early stopping @ epoch {epoch}")
                    break
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train_loss {avg_train_loss:.4f} | "
              f"train_cont: {avg_train_cont:.4f} | "
              f"train_recon: {avg_train_recon:.4f} | "
              f"val_loss {avg_val_loss:.4f} | "
              f"val_cont: {avg_val_cont:.4f} | "
              f"val_recon: {avg_val_recon:.4f} | "
              f"lr: {current_lr:.2e}")
        
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "train_cont_loss": avg_train_cont,
            "train_recon_loss": avg_train_recon,
            "val_loss": avg_val_loss,
            "val_cont_loss": avg_val_cont,
            "val_recon_loss": avg_val_recon,
            "learning_rate": current_lr,
        })
        
        img2txt_acc, txt2img_acc, easy, medium = evaluate_retrieval(img_ae, txt_ae, test_loader)
        record = dict(epoch=epoch, img2txt_acc=img2txt_acc, txt2img_acc=txt2img_acc, easy=easy, medium=medium)
        epoch_records.append(record)
    
    print("Training complete!")
    return epoch_records

def save_records_csv(records, csv_path):
    if not records:
        print("WARNING: nothing to save")
        return
    dirpath = os.path.dirname(csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Results saved to {csv_path!r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Training Script for Joint AE Bridge")
    parser.add_argument("--mode", choices=["spread", "baseline_con", "baseline_negcon"], required=True, help="Training mode: spread, con, or negcon")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image embeddings .pkl file")
    parser.add_argument("--positive_text_path", type=str, required=True, help="Path to positive text embeddings .pkl file")
    parser.add_argument("--negative_text_path", type=str, required=True, help="Path to negative text embeddings .pkl file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--shuffle_seed", type=int, default=55)
    parser.add_argument("--text_agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension for autoencoders")
    parser.add_argument("--hidden_dim", default=(512, 512, 512), help="Hidden dimensions for autoencoders")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--csv_out", type=str, default="./jam_output_results.csv")
    parser.add_argument("--max_alpha", type=float, default=0.5, help="Alpha value for spread loss (used in fixed mode)")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = load_data(
        args.image_path, args.positive_text_path, args.negative_text_path,
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        shuffle_seed=args.shuffle_seed,
        text_agg=args.text_agg,
    )

    wandb.init(project="jam_output", name=os.path.basename(args.csv_out).replace(".csv", ""))
    
    if args.mode == "spread":
        epoch_records = train_model_spread(
            train_loader, val_loader, test_loader,
            max_alpha=args.max_alpha,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dim,
            epochs=args.epochs,
        )
    elif args.mode == "baseline_con":
        epoch_records = train_model_baseline(
            train_loader, val_loader, test_loader,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dim,
            align_type="lcon",
            epochs=args.epochs,
        )
    elif args.mode == "baseline_negcon":
        epoch_records = train_model_baseline(
            train_loader, val_loader, test_loader,
            latent_dim=args.latent_dim,
            hidden_dims=args.hidden_dim,
            align_type="lconneg",
            epochs=args.epochs,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    wandb.finish()
    
    save_records_csv(epoch_records, args.csv_out)
