import pandas as pd
import numpy as np

import sys 
sys.path.append("../")
import numpy as np 
import os 

try:
    from apex_oracle import apex_wrapper
except:
    assert 0

def process_in_chunks(data, chunk_size, process_function):
    # Process the data in chunks and stack results
    num_items = len(data)
    results = []
    for start in range(0, num_items, chunk_size):
        end = start + chunk_size
        chunk = data[start:end]
        result = process_function(chunk)
        results.append(result)  # Collect each chunk's result
    # Stack all results vertically
    return np.vstack(results)

from apex_oracle import apex_wrapper

# Load data
file_path = '../apex_oracle/results/templates.txt'
output_path = '../apex_oracle/results/templates_mics.csv'
x_list = pd.read_csv(file_path, header=None)[0].to_list()

# Process data in chunks of 100 sequences
chunk_size = 100
scores = process_in_chunks(x_list, chunk_size, apex_wrapper)
np.savetxt(output_path, scores, delimiter=",", fmt="%.8f")

print("Scores shape:", scores.shape)
print("Scores:", scores)
