''' Define your objecive function(s) here
Note: All code assumes we seek to maximize f(x)
If you want to instead MINIMIZE the objecitve, multiple scores by -1 in 
your query_black_box() method 
''' 

import sys 
sys.path.append("../")
import numpy as np 
import os 

try:
    from apex_oracle import apex_wrapper
except:
    assert 0

import sys
import os
from Bio import SeqIO

sys.path.append("../../../")

from oracle.inference_oracle import inference_oracle

class ObjectiveFunction:
    ''' Objective function f, we seek x that MAXIMIZE f(x)'''
    def __init__(self,):
        pass
    
    def __call__(self, x_list):
        ''' Input 
                x_list: 
                    a LIST of input space items from the origianl input 
                    search space (i.e. list of aa seqs)
            Output 
                scores_list: 
                    a LIST of float values obtained by evaluating your 
                    objective function f on each x in x_list
                    or np.nan in the wherever x is an invalid input 
        '''
        return self.query_black_box(x_list)


    def query_black_box(self, x_list):
        ''' Input 
                x_list: a list of input space items x_list
            Output 
                scores_list: 
                    a LIST of float values obtained by evaluating your 
                    objective function f on each x in x_list
                    or np.nan in the wherever x is an invalid input 
        '''
        raise NotImplementedError("Must implement method query_black_box() for the black box objective")


class ExampleObjective(ObjectiveFunction):
    ''' Example objective funciton length of the input space items
        This is just a dummy example where the objective is the 
        avg number of A and M's in the sequence 
    ''' 
    def __init__(self,):
        super().__init__()

    def query_black_box(self, x_list):
        scores_list = []
        for x in x_list:
            if type(x) != str:
                score = np.nan 
            elif len(x) == 0:
                score = 0 
            else:
                score = 0 
                for char in x: 
                    if char in ["A", "M"]:
                        score += 1
                score = score/len(x)
            scores_list.append(score)

        return scores_list 


class ApexObjective(ObjectiveFunction):
    ''' Objective function for optimizing against the provided APEX Oracle
    ''' 
    def __init__( 
        self,
        score_version="mean",
    ):
        self.score_version = score_version

        self.cache = {}

        super().__init__()


    def query_black_box(self, x_list):
        if len(x_list) == 0: 
            return [] 
        
        # Trim each sequence to max of 150 characters to make oracle happy
        x_list = [x[:150] for x in x_list]

        # collect unseen sequences
        unseen_sequences = [x for x in x_list if x not in self.cache]

        # Query the oracle only for unseen sequences
        if unseen_sequences:
            new_scores = apex_wrapper(unseen_sequences)
            
            # Update the cache with new results
            for x, score in zip(unseen_sequences, new_scores):
                self.cache[x] = score

        # Retrieve scores in the original order
        scores = [self.cache[x] for x in x_list]
        # turn scores into a numpy array
        scores = np.array(scores)

        if self.score_version == "mean":
            # average MIC across all bacteria
            scores = - scores.mean(axis=1)
        elif self.score_version == "min":
            # lowest MIC across all bacteria
            scores = - scores.max(axis=1)
        elif self.score_version == "max":
            # max MIC across all bacteria
            scores = - scores.min(axis=1)
        elif self.score_version == "gramneg_mean":
            # optimize first 7 entries of the score vector, which correspond to gram-negative bacteria
            scores = - scores[:, :7].mean(axis=1)
        elif self.score_version == "grampos_mean":
            # optimize last 4 entries of the score vector, which correspond to gram-positive bacteria
            scores = - scores[:, 7:].mean(axis=1)
        elif self.score_version == "gramnegonly_mean":
            # optimize first 7 entries of the score vector, while penalizing the last 4 entries by a factor of 10 to make sure we are not optimizing for them
            # the penalty is applied only if the MIC value is smaller than 32, signalling antimicrobial activity
            penalty = 100
            scores = - scores[:, :7].mean(axis=1) - penalty * (32 - scores[:, 7:]).clip(min=0).sum(axis=1)
        elif self.score_version == "gramposonly_mean":
            # optimize last 4 entries of the score vector, while penalizing the first 7 entries by a factor of 10 to make sure we are not optimizing for them
            # the penalty is applied only if the MIC value is smaller than 32, signalling antimicrobial activity
            penalty = 100
            scores = - scores[:, 7:].mean(axis=1) - penalty * (32 - scores[:, :7]).clip(min=0).sum(axis=1)
        else:
            assert 0 

        return scores.tolist()

class SGPOObjective(ObjectiveFunction):
    ''' Objective function for optimizing against oracles used in SGPO
    ''' 
    def __init__( 
        self,
        protein="TrpB",
    ):
        self.protein = protein
        self.model_path = f"../../../oracle/checkpoints/{protein}"
        self.full_seq = SeqIO.read(f"../../../data/{protein}/parent.fasta", "fasta").seq

        super().__init__()


    def query_black_box(self, x_list, impose_penalty=True):
        if self.protein == "TrpB":
            self.residues = [117, 118, 119, 162, 166, 182, 183, 184, 185, 186, 227, 228, 230, 231, 301]
            #for every x in x_list, subsample only the residues that are mutated, using 0 indexing
            for j, x in enumerate(x_list):
                x = list(x)
                x_subsample = [x[i-1] for i in self.residues]
                x_list[j] = ''.join(x_subsample)

        _, scores = inference_oracle(x_list, self.protein, self.model_path, impose_penalty=impose_penalty, full_seq=self.full_seq)

        return scores.tolist()

'''Objective functions with unique string identifiers 
identifiers can be passed in when running LOL-BO or ROBOT with --task_id arg
whcih specifies which diversity function to use 
--task_specific_args can be used to specify a list of args passed into the init of 
any of these objectives when they are initialized 
'''
OBJECTIVE_FUNCTIONS_DICT = {
    'example':ExampleObjective,
    'apex':ApexObjective,
    'sgpo':SGPOObjective,
}
