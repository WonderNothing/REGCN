import os
import pandas as pd
import numpy as np
dataset = 'SSE'
# Define the input and output folder paths
input_folder = '../data/data/VMDdata/'+ dataset+'/' # Replace with the path to the input folder containing CSV files
output_folder = '../data/data/VMDnor/'+dataset+'/'  # Replace with the path to the output folder to save the modified files

# Get a list of CSV file names in the input folder
file_names = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Normalize each column of CSV files and save modified files
for file_name in file_names:
    # Read the CSV file
    file_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(file_path)

    # Normalize each column
    min_vals = df.min()
    max_vals = df.max()
    normalized_df = (df - min_vals) / (max_vals - min_vals)

    # Insert min and max values at the end of each column
    normalized_df = normalized_df.append([min_vals, max_vals], ignore_index=True)

    # Save the modified file with the same filename in the output folder
    output_path = os.path.join(output_folder, file_name)
    normalized_df.to_csv(output_path, index=False)

print("Normalization and modification completed.")