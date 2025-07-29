import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import sys
import os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{file_dir}")
from APEX_models import AMP_model
from utils import *


pathogen_list = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIG221",
    "E. coli AIG222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]

max_len = 52  # maximum seq length; 52 = start character + maximum peptide length (50 aa) + end character; longer peptides will be truncated
word2idx, idx2word = make_vocab()  # make amino acid vocabulary
# emb, AAindex_dict = AAindex('./aaindex1.csv', word2idx) #make amino acid embeddings


# Load pretrained APEX models (8 in total)
APEX_models = []
file_dir = os.path.dirname(os.path.abspath(__file__))
for a_model in glob.glob(f"{file_dir}/APEX_pathogen_models/APEX_*"):
    model = torch.load(a_model)
    model.eval()
    APEX_models.append(model)


batch_size = 3000  # change according to your GPU memory


# Use pretrained APEX models to predict species-specific antimicrobial activity (i.e., minimum inhibitory concentration [MIC]; unit: uM)
# 8 pretrained APEX models are provided, and predictions are averaged
def predict_APEX(seq_list):
    for ensemble_id in range(len(APEX_models)):
        AMP_model = APEX_models[ensemble_id].cuda().eval()

        data_len = len(seq_list)
        for i in range(int(math.ceil(data_len / float(batch_size)))):
            seq_batch = seq_list[i * batch_size : (i + 1) * batch_size]
            seq_rep = onehot_encoding(seq_batch, max_len, word2idx)  # make input
            X_seq = torch.LongTensor(seq_rep).cuda()

            AMP_pred_batch = AMP_model(X_seq).cpu().detach().numpy()  # make predictions
            AMP_pred_batch = 10 ** (
                6 - AMP_pred_batch
            )  # transform back to MICs; When training the APEX models, MICs were transformed by: -np.log10(MICs/float(1000000))

            if i == 0:
                AMP_pred = AMP_pred_batch
            else:
                AMP_pred = np.vstack([AMP_pred, AMP_pred_batch])

        # sum up the predictions made by different APEX models
        if ensemble_id == 0:
            AMP_sum = AMP_pred
        else:
            AMP_sum += AMP_pred

    AMP_pred = AMP_sum / float(len(APEX_models))  # average the predictions

    return AMP_pred


# AMP_pred = predict_APEX(seq_list)

# df = pd.DataFrame(data=AMP_pred, columns=pathogen_list, index=seq_list)
# #print (df)

# #save the prediction result
# df.to_csv('Predicted_MICs.csv')

apex_wrapper = predict_APEX
apex_best_wrapper = lambda x: np.min(apex_wrapper(x), axis=1)
apex_mean_wrapper = lambda x: np.mean(apex_wrapper(x), axis=1)
apex_worst_wrapper = lambda x: np.max(apex_wrapper(x), axis=1)
