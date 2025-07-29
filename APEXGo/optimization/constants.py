import os
""" Constants used by optimization routine """

REFERENCE_SEQUENCE = [
    "RACLHARSIARLHKRWRPVHQGLGLK",
    "KTLKIIRLLF",
    "KRKRGLKLATALSLNNKF",
    "KIYKKLSTPPFTLNIRTLPKVKFPK",
    "RMARNLVRYVQGLKKKKVI",
    "RNLVRYVQGLKKKKVIVIPVGIGPHANIK",
    "CVLLFSQLPAVKARGTKHRIKWNRK",
    "GHLLIHLIGKATLAL",
    "RQKNHGIHFRVLAKALR",
    "HWITINTIKLSISLKI",
]

file_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_VAE_STATE_DICT = f"{file_dir}/uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
ENCODER_DIM = 256
DECODER_DIM = 256
KL_FACTOR = 0.0001
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6


PATH_TO_INITILIZATION_DATA = None

ALL_AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
