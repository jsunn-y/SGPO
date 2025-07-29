# APEX pathogen - predicting species-specific antimicrobial activity against pathogens using deep learning

## Predict AMPs using APEX
By running APEX_predict.py, species-specific antmicrobial activties (MICs) of peptides in test_seqs.txt will be generated and saved in Predicted_MICs.csv.


Pathogen list: 

A. baumannii ATCC 19606	 

E. coli ATCC 11775	

E. coli AIG221	

E. coli AIG222	

K. pneumoniae ATCC 13883	

P. aeruginosa PA01	

P. aeruginosa PA14	

S. aureus ATCC 12600	

S. aureus (ATCC BAA-1556) - MRSA	

vancomycin-resistant E. faecalis ATCC 700802	

vancomycin-resistant E. faecium ATCC 700221


## File description
Folder APEX_pathogen_models contains the pretrained APEX models (8 in total)

APEX_models.py: defines the neural network architecture of APEX

utils.py: helper functions

aaindex1.csv: amino acid embeddings (freezed during training, obtained from https://www.genome.jp/aaindex/)

test_seqs.txt: example input of APEX_predict.py; each line is a peptide sequence (<= 50 amino acids)

Predicted_MICs.csv: example output of APEX_predict.py

APEX_predict.py: predict species-specific antmicrobial activties (minimum inhibitory concentration [MIC]; unit: uM) against 11 pathogens for peptides in test_seqs.txt. We use 8 APEX models trained under different neural network architectures or training strategies to make predictions. Predictions from the base learners will be averaged to generate the final activity prediction. 



## Software version
pytorch: 1.11.0+cu113


## Contacts
If you have any questions or comments, please feel free to email Fangping Wan (fangping[dot]wan[at]pennmedicine[dot]upenn[dot]edu) and/or César de la Fuente (cfuente[at]pennmedicine[dot]upenn[dot]edu).

