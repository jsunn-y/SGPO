import sys 
sys.path.append("../")
from uniref_vae.transformer_vae_unbounded import InfoTransformerVAE 
from uniref_vae.data import DataModuleKmers
import torch 
from constants import (
    ENCODER_DIM,
    DECODER_DIM,
    KL_FACTOR,
    ENCODER_NUM_LAYERS,
    DECODER_NUM_LAYERS,
)

# example function to load vae, loads uniref vae 
def load_vae(
    path_to_vae_statedict,
    dim=1024, # dim//2
    max_string_length=150,
):
    data_module = DataModuleKmers(
        batch_size=10,
        k=1,
        load_data=False,
    )
    dataobj = data_module.train
    vae = InfoTransformerVAE(
        dataset=dataobj, 
        d_model=dim//2,
        kl_factor=KL_FACTOR,
        encoder_dim_feedforward=ENCODER_DIM,
        decoder_dim_feedforward=DECODER_DIM,
        encoder_num_layers=ENCODER_NUM_LAYERS,
        decoder_num_layers=DECODER_NUM_LAYERS,
    ) 

    # load in state dict of trained model:
    if path_to_vae_statedict:
        state_dict = torch.load(path_to_vae_statedict) 
        vae.load_state_dict(state_dict, strict=True) 
    vae = vae.cuda()
    vae = vae.eval()

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj 
