import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random
import wandb
from Bio import SeqIO

def set_seed(seed):
    random.seed(seed)                  # Python random module
    np.random.seed(seed)               # NumPy random seed
    torch.manual_seed(seed)            # PyTorch CPU seed
    torch.cuda.manual_seed(seed)       # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)   # All GPUs seed (if using multi-GPU)

class OracleDataset(Dataset):
    def __init__(self, protein, split=None):
        if split in ["train", "validation", "test"]:
            df = pd.read_csv(f"data/{protein}/fitness.csv")
            df = df[df["split"] == split]
            self.sequences = df["Combo"].values
            self.X = torch.tensor(np.array([self.get_onehot_encoding(seq) for seq in self.sequences])).float()
            self.y = torch.tensor(df["fitness"].values).float().reshape(-1, 1)
        elif isinstance(split, str) or isinstance(split, list):
            #fasta
            if isinstance(split, str):
                seqs = list(SeqIO.parse(split, "fasta"))
                seqs = [str(seq.seq) for seq in seqs]
            else:
                seqs = split
            self.sequences = seqs
            self.X = torch.tensor(np.array([self.get_onehot_encoding(seq) for seq in seqs])).float()
            self.y = torch.zeros(len(seqs))

    def get_onehot_encoding(self, sequence):
        # Example one-hot encoding function 
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        # alphabet = "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!"
        encoding = np.zeros((len(sequence), len(alphabet)))
        for i, aa in enumerate(sequence):
            encoding[i, alphabet.index(aa)] = 1
        #flatten encoding
        encoding = encoding.flatten()
        return encoding

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sequences[idx]

# Model architecture and hyperparameters modified from Blalock et al. "Functional Alignment of Protein Language Models via Reinforcement Learning with Experimental Feedback"

# Define the MLP model
class OracleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, dropout_rate=0.1):
        super(OracleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

LEARNING_RATE = 1e-6
BATCH_SIZE = 128
EPOCHS = 50 #2000 in the original paper, needs this many to learn something meaningful, use 1000 for TrpB which has 20 times more data, use even fewer (50) for GB1
DROPOUT = 0.1
PATIENCE = 400
HIDDEN_DIM = 400
ACTIVATION = 'ReLU'
OPTIMIZER = 'Adam'
LOSS_FN = 'MSE'  # Mean Squared Error
EMBEDDING_TYPE = 'One-Hot'  # You would need to preprocess your data accordingly
N_ENSEMBLE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.MSELoss()


if __name__ == "__main__":
    protein = "GB1" #"TrpB" #CreiLOV
    # Prepare data loaders
    # Could also consider bootstrap ensembling instead of training on the same data
    train_dataset = OracleDataset(protein=protein, split="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset = OracleDataset(protein=protein, split="validation")
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_dataset.X.shape[1]
    save_path = "oracle/checkpoints/" + protein
    os.makedirs(save_path, exist_ok=True)

    for ensemble_idx in range(N_ENSEMBLE):
        print(f"Training ensemble member {ensemble_idx + 1}...")
        wandb.init(project="protein-fitness-oracle", name=f"ensemble_{ensemble_idx + 1}")

        #set seed
        set_seed(ensemble_idx)

        # Initialize model, loss function, and optimizer
        model = OracleModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        best_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, targets, _ in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                wandb.log({"loss": loss})
            #print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")
            wandb.log({"epoch_loss": running_loss/len(train_loader)})

            avg_loss = running_loss / len(train_loader)

            # Early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Logging
            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                #print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets, _ in validation_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss /= len(validation_loader)
                wandb.log({"val_loss": val_loss})

        wandb.finish() 
        
        # Save the model
        torch.save(model.state_dict(), f"{save_path}/model{ensemble_idx}.pth")
        print("Training complete. Model saved.")

    print("Training complete.")