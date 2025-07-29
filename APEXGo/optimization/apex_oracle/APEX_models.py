import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class PeptideEmbeddings(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.aa_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb), padding_idx=0)
    def forward(self, x):
        out = self.aa_embedding(x)
        return out

class AMP_model(nn.Module):
    def __init__(self, emb, emb_size, num_rnn_layers, dim_h, dim_latent, num_fc_layers, num_task):
        super().__init__()

        self.peptideEmb = PeptideEmbeddings(emb=emb)
        self.dim_emb = emb_size
        self.dim_h = dim_h
        self.dropout = 0.1
        self.dim_latent = dim_latent
        max_len = 52

        self.rnn = nn.GRU(emb_size, dim_h, num_layers=num_rnn_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.layernorm = nn.LayerNorm(dim_h * 2)
        self.attn1 = nn.Linear(dim_h * 2 + emb_size, max_len)
        self.attn2 = nn.Linear(dim_h * 2, 1)

        self.fc0 =  nn.Linear(dim_h * 2, dim_h)

        self.fc1 = nn.Linear(dim_h, dim_latent)
        self.fc2 = nn.Linear(dim_latent, int(dim_latent / 2))
        self.fc3 = nn.Linear(int(dim_latent / 2), int(dim_latent / 4))
        self.fc4 = nn.Linear(int(dim_latent / 4), num_task)

        self.ln1 = nn.LayerNorm(dim_latent) 
        self.ln2 = nn.LayerNorm(int(dim_latent / 2)) 
        self.ln3 = nn.LayerNorm(int(dim_latent / 4)) 

        self.dp1 = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.1)#nn.Dropout(0.2)



        self.fc1_ = nn.Linear(dim_h, dim_latent)
        self.fc2_ = nn.Linear(dim_latent, int(dim_latent / 2))
        self.fc3_ = nn.Linear(int(dim_latent / 2), int(dim_latent / 4))
        self.fc4_ = nn.Linear(int(dim_latent / 4), 1)

        self.ln1_ = nn.LayerNorm(dim_latent) 
        self.ln2_ = nn.LayerNorm(int(dim_latent / 2)) 
        self.ln3_ = nn.LayerNorm(int(dim_latent / 4)) 

        self.dp1_ = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp2_ = nn.Dropout(0.1)#nn.Dropout(0.2)
        self.dp3_ = nn.Dropout(0.1)#nn.Dropout(0.2)




    def forward(self, x):

        x = self.peptideEmb(x)
        #h = self.initH(x.shape[0])
        #out, h = self.rnn(x, h)
        out, h = self.rnn(x)
        out = self.layernorm(out)

        attn_weights1 = F.softmax(self.attn1(torch.cat((out, x), 2)), dim=2) #to be tested: masked softmax
        #attn_weights1.permute(0, 2, 1)
        out = torch.bmm(attn_weights1, out)
        attn_weights2 = F.softmax(self.attn2(out), dim=1) #to be tested: masked softmax
        out = torch.sum(attn_weights2 * out, dim=1) #to be test: masked sum

        out = self.fc0(out)

        out = self.dp1(F.relu(self.ln1(self.fc1(out))))
        out = self.dp2(F.relu(self.ln2(self.fc2(out))))
        out = self.dp3(F.relu(self.ln3(self.fc3(out))))
        out = self.fc4(out)

        return F.relu(out)

    def predict(self, x):
        return self.forward(x)


    def clf_forward(self, x):

        x = self.peptideEmb(x)
        #h = self.initH(x.shape[0])
        #out, h = self.rnn(x, h)
        out, h = self.rnn(x)
        out = self.layernorm(out)

        attn_weights1 = F.softmax(self.attn1(torch.cat((out, x), 2)), dim=2) #to be tested: masked softmax
        #attn_weights1.permute(0, 2, 1)
        out = torch.bmm(attn_weights1, out)
        attn_weights2 = F.softmax(self.attn2(out), dim=1) #to be tested: masked softmax
        out = torch.sum(attn_weights2 * out, dim=1) #to be test: masked sum

        out = self.fc0(out)

        out = self.dp1_(F.relu(self.ln1_(self.fc1_(out))))
        out = self.dp2_(F.relu(self.ln2_(self.fc2_(out))))
        out = self.dp3_(F.relu(self.ln3_(self.fc3_(out))))
        out = self.fc4_(out)

        return out
