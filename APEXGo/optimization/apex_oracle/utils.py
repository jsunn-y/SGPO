import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from torch.autograd import Variable
from scipy import stats
import pandas as pd 
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split
import os.path

def make_vocab():
    #0: pad
    #1: start
    #2: end

    word2idx = {}
    idx2word = {}

    word2idx['0'] = 0
    word2idx['1'] = 1
    word2idx['2'] = 2

    word2idx['A'] = 3
    word2idx['C'] = 4
    word2idx['D'] = 5
    word2idx['E'] = 6
    word2idx['F'] = 7
    word2idx['G'] = 8
    word2idx['H'] = 9
    word2idx['I'] = 10
    word2idx['K'] = 11
    word2idx['L'] = 12
    word2idx['M'] = 13
    word2idx['N'] = 14
    word2idx['P'] = 15
    word2idx['Q'] = 16
    word2idx['R'] = 17
    word2idx['S'] = 18
    word2idx['T'] = 19
    word2idx['V'] = 20
    word2idx['W'] = 21
    word2idx['Y'] = 22

    for key, value in word2idx.items():
        idx2word[value] = key

    return word2idx, idx2word


def AAindex(path, word2idx):
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        AAindex_dict = {}
        AAindex_matrix = []
        skip = 1
        for row in reader:
            if skip == 1:
                skip = 0
                header = np.array(row)[1:].tolist()
                continue
            tmp = []
            for j in np.array(row)[1:]:
                try:
                    tmp.append(float(j))
                except:
                    tmp.append(0)
            AAindex_matrix.append(np.array(tmp))

        dim = np.shape(AAindex_matrix)[0]
        AAindex_matrix = np.array(AAindex_matrix)
        for i in range(len(header)):
            AAindex_dict[header[i]] = AAindex_matrix[:, i]

    #print (AAindex_matrix)
    emb = np.zeros((len(word2idx), dim))
    for key, value in word2idx.items():
        if key in AAindex_dict:
            emb[value] = AAindex_dict[key]
        else:
            pass
    return emb, AAindex_dict



def onehot_encoding(seq_list_, max_len, word2idx):
    #0: pad
    #1: start
    #2: end
    seq_list = [i for i in seq_list_]
    X = np.zeros((len(seq_list), max_len)).astype(int)

    AA_mask = []
    nonAA_mask = []

    for i in range(len(seq_list)):
        if len(seq_list[i]) >= max_len - 2:
            a_seq = '1' + seq_list[i][:max_len-2].upper() + '2'
        else:
            a_seq = '1' + seq_list[i].upper() + '2'

        if len(a_seq) > max_len:
            iter_num = max_len
        else:
            iter_num = len(a_seq)

        for j in range(iter_num):
            if a_seq[j] not in word2idx:
                continue
            else:
                X[i,j] = word2idx[a_seq[j]]

        tmp = np.zeros(max_len)
        tmp[1:iter_num+1] = 1
        AA_mask.append(tmp.astype(int))
        nonAA_mask.append((1-tmp).astype(int))


    return np.array(X)#, np.array(AA_mask), np.array(nonAA_mask)
