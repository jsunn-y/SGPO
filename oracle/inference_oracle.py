import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from Bio import SeqIO

from oracle.train_oracle import OracleDataset, OracleModel, BATCH_SIZE, HIDDEN_DIM, DROPOUT, DEVICE, criterion

def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def penalty(hamming_distances, cutoff=5, rate=0.95):
    if hamming_distances <= cutoff:
        return 1
    else:
        return rate ** (hamming_distances - cutoff)

def inference_oracle(split, protein, model_path, impose_penalty=True):
    """
    impose_penalty: whether or not to penalize the fitness of sequences that are too different from the full sequence
    """
    test_dataset = OracleDataset(protein=protein, split=split) #accepts split as a string, or list of strings
    full_seq = SeqIO.read(f"data/{protein}/parent.fasta", "fasta").seq
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = test_dataset.X.shape[1]
    #find the number of files in the directory
    files = os.listdir(model_path)
    all_predictions = np.zeros((len(files), len(test_dataset)))

    for i, file in enumerate(files):
        model = OracleModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT)
        model.load_state_dict(torch.load(f"{model_path}/{file}"))
        model.to(DEVICE)
        
        model.eval()
        test_loss = 0.0
        predictions_list = []

        with torch.no_grad():
            for inputs, targets, sequences in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                predictions = outputs.cpu().numpy()

                #added this to penalize predictions that are too different from the full sequence (once it's >60% different)
                if impose_penalty:
                    if protein == "CreiLOV":
                        cutoff = 70
                        rate = 0.99
                    elif protein == "TrpB":
                        cutoff = 233 #shouldn't occur
                        rate = 0.99
                    elif protein == "GB1":
                        cutoff = 33 
                        rate = 0.99
                    #get the hamming distances between the sequences and the full sequence
                    hamming_distances = [hamming_distance(full_seq, seq) for seq in sequences]
                    #print(predictions)
                    #apply penalty to predictions
                    predictions = predictions.reshape(-1) * np.array([penalty(hd, cutoff=cutoff, rate=rate) for hd in hamming_distances])
                
                #round up negative predictions to zero (there shouldn't be any for CreiLOV)
                predictions = np.maximum(predictions, 0)
                predictions_list.append(predictions)

        predictions = np.concatenate(predictions_list).reshape(-1)
        all_predictions[i] = predictions
    
    predictions = torch.tensor(np.mean(all_predictions, axis=0))
    test_loss = criterion(torch.tensor(predictions), test_dataset.y)
    test_loss /= len(test_loader)
    
    return test_loss, predictions #test_loss is meaningless if split is not in train, validation, or test

if __name__ == "__main__":
    test_loss, predictions = inference_oracle("test", "CreiLOV", "oracle/checkpoints/CreiLOV")
    print(f"Test loss: {test_loss}")
    print(predictions)