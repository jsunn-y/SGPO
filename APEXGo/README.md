
This is a modified version of the original APEXGo repository. For more information, visit the [orginial repository](https://github.com/Yimeng-Zeng/APEXGo).

## Installation:
For a local setup using conda, run the following:
```shell
conda create --name apexgo python=3.10
conda activate apexgo
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install tqdm==4.65.0 wandb==0.18.6 botorch==0.12.0 selfies==2.1.2 guacamol==0.5.5 rdkit==2024.3.6 lightning==2.4.0 joblib==1.4.2 fire==0.7.0 levenshtein==0.26.1 rotary_embedding_torch==0.8.5 gpytorch==1.13 pandas==2.2.3 numpy==1.24.3 fcd_torch==1.0.7 matplotlib==3.9.2
```

## VAE Training:
To run VAE training, replace `generation/data/uniref-cropped.csv` with the sequences you want to train with, in the same format. Then customize `generation/train.sh` and run:
```
bash generation/train.sh
```
To check the qualty of generated sequences, use `generation/generation.ipynb` and `generation/generation.py`. The model checkpoints will be saved to `generation/saved_models`.

## Optimization with APEXGo:
To run the APEXGo algorithm and save outputed sequences with associated fitness values, customize `optimization/constrained_bo_scripts/optimize.sh` and run:
```
bash optimization/constrained_bo_scripts/optimize.sh
```
The output will be automatically saved to the corresponding experimental output directory in SGPO.



