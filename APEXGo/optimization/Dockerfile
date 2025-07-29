FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y wget git
RUN pip install tqdm wandb botorch selfies guacamol rdkit lightning joblib fire levenshtein rotary_embedding_torch gpytorch pandas numpy fcd_torch matplotlib

ENV WANDB_API_KEY=YOUR_API_KEY
WORKDIR /workspace