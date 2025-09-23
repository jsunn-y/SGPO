from load_vae import *
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

vae, dataobj  = load_vae(
    "saved_models/smart-lion-4/smart-lion-4_model_state_epoch_103.pkl",
    dim=256, # dim//2
    max_string_length=400,
)

df = pd.read_csv("/disk1/jyang4/repos/APEXGo/generation/data/uniref-cropped.csv")
xs_batch = list(df["sequence"][:1000].values)
z, vae_loss = vae_forward(xs_batch, dataobj, vae)

decoded_seqs = vae_decode(z, vae, dataobj)

parent_seq = "MKGYFGPYGGQYVPEILMGALEELEAAYEGIMKDESFWKEFNDLLRDYAGRPTPLYFARRLSEKYGARVYLKREDLLHTGAHKINNAIGQVLLAKLMGKTRIIAETGAGQHGVATATAAALFGMECVIYMGEEDTIRQKLNVERMKLLGAKVVPVKSGSRTLKDAIDEALRDWITNLQTTYYVFGSVVGPHPYPIIVRNFQKVIGEETKKQIPEKEGRLPDYIVACVSGGSNAAGIFYPFIDSGVKLIGVEAGGEGLETGKHAASLLKGKIGYLHGSKTFVLQDDWGQVQVSHSVSAGLDYSGVGPEHAYWRETGKVLYDAVTDEEALDAFIELSRLEGIIPALESSHALAYLKKINIKGKVVVVNLSGRGDKDLESVLNHPYVRERIR"

amino_acids = "ACDEFGHIKLMNPQRSTVWY"
#check for special non-amino acid characters and replace with identity in parent sequence
for i, seq in enumerate(decoded_seqs):
    seq_list = list(seq)

    for j, char in enumerate(seq):
        if char not in amino_acids:
            #convert each decoded sequence to a list
            #replace the character with the corresponding character in the parent sequence
            seq_list[j] = parent_seq[j]
            #convert back to string
            
    decoded_seqs[i] = ''.join(seq_list)

#save the list to fasta

records = [SeqRecord(seq=seq, id=str(i), description="") for i, seq in enumerate(decoded_seqs)]
SeqIO.write(records, "data/generated.fasta", "fasta")

