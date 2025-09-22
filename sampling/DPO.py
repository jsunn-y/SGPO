from .base import Algo
import torch
import tqdm
import torch.nn.functional as F
import numpy as np
import os
from Bio import SeqIO

from models.causalLM import CausalLM
from models.pretraining.model.progen2.model import ProGenForCausalLM
from models.pretraining.model.progen2.tokenizer import get_tokenizer

class DPOInpaint(Algo):
    def __init__(self, net, data_config, n_max_mutations=None, device='cuda', temp=1, top_p=0.95):
        super().__init__(net=net, data_config=data_config, n_max_mutations=n_max_mutations, device=device, causal_LM=True)
        #language model-specific parameters
        self.temp = temp
        self.top_p = top_p
        self.tokenizer = get_tokenizer()
        self.name = data_config.name
    
    def update_model(self, net):
        self.net = net

    # def project(self, x):
    #     #TODO: update this function
    #     return x * self.mask + self.full_seq * (1 - self.mask)

    def project_sequences(self, sequences):
        """
        Aligns sequences to a reference and fills in gaps with the reference sequence. Also replaces all non-mutated residues with the reference sequence.
        """
        #save to fasta
        #if sequences is a string of the fasta file name
        if type(sequences) == str:
            os.system(f"mafft --quiet --add {sequences} --keeplength data/{self.name}/parent.fasta > tmp/aligned.fasta")
        else:
            #make tmp directory if it does not exist
            os.makedirs(f"tmp/{self.name}", exist_ok=True)
            with open(f"tmp/{self.name}/temp.fasta", "w") as f:
                for i, seq in enumerate(sequences):
                    f.write(f">{i}\n")
                    f.write(f"{seq}\n")
            os.system(f"mafft --quiet --add tmp/{self.name}/temp.fasta --keeplength data/{self.name}/parent.fasta > tmp/{self.name}/aligned.fasta")

        aligned = list(SeqIO.parse(f"tmp/{self.name}/aligned.fasta", "fasta")) #cannot run two campaigns with the same protein name at the same time

        #alternatively return parent seq repeated - might not be ideal if the entire mafft run failed, but this shows that the generations are not great
        # if len(aligned) == 0:
        #     aligned = len(sequences) * [self.full_seq]

        aligned = [str(s.seq) for s in aligned]
        #replace gaps with the parent sequence
        for i, seq in enumerate(aligned):
            for j, r in enumerate(seq):
                if r == "-":
                    # sample a random residue from aa_options
                    # random_res = random.choice(aa_options)
                    # aligned[i] = aligned[i][:j] + random_res + aligned[i][j+1:]
                    aligned[i] = aligned[i][:j] + self.full_seq[j] + aligned[i][j+1:] #alteratively fill with WT
            #convert to list of strings

        if self.n_residues > 4:
            if self.n_max_mutations is not None:
                #choose n_max_mutations residues and limit mutations to those positions
                for i, seq in enumerate(aligned):
                    #get indices of mutated residues by comparing seq to the reference
                    mutated_residues = [j for j, (r1, r2) in enumerate(zip(seq, self.full_seq)) if r1 != r2]
                    mutated_residues = [j for j in mutated_residues if j in self.residues] #limit to the mutated positions
                    #choose n_max_mutations residues and limit mutations to those positions
                    max_mutations = min(self.n_max_mutations, len(mutated_residues))
                    mutated_residues = np.random.choice(mutated_residues, max_mutations, replace=False)
                    # mutated_residues = np.random.choice(self.residues, self.n_max_mutations, replace=False)
                    seq = list(seq)
                    for j, r in enumerate(seq):
                        if j not in mutated_residues:
                            seq[j] = self.full_seq[j]
                    aligned[i] = ''.join(seq)
            else:
                #replace non-mutated residues with the parent sequence
                for i, seq in enumerate(aligned):
                    seq = list(seq)
                    for j, r in enumerate(seq):
                        if j not in self.residues:
                            seq[j] = self.full_seq[j]
                    aligned[i] = ''.join(seq)

        #replace special tokens with a random one in ACDEFGHIKLMNPQRSTVWY
        for i, seq in enumerate(aligned):
            seq = list(seq)
            for j, r in enumerate(seq):
                if r not in list("ACDEFGHIKLMNPQRSTVWY"):
                    seq[j] = np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
            aligned[i] = ''.join(seq)

        return aligned
    
    def inference(self, num_samples=1, verbose=True, detokenize=True, project=True):
        """
        Inference for a single batch.
        """
        assert type(self.net) == CausalLM
        detokenized = self.net.sample(num_return_sequences=num_samples, temperature=self.temp, top_p=self.top_p)
        if project:
            detokenized = self.project_sequences(detokenized) #align to reference and fill in gaps (project)

        return None, detokenized #currently does not support tokenized samples

