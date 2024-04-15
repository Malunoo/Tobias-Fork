import json
import pandas as pd

# Path to the text file containing the .jsonl file path
file_path_text_file = r'C:\Users\Brandon\Documents\Repositorys\MPTJB\MPTJB\AI\jsonl_file_path.txt'

# Read the .jsonl file path from the text file
with open(file_path_text_file, 'r', encoding='utf-8') as f:
    jsonl_file_path = f.readline().strip()

# Dictionary to hold column names and a list of their sample values
columns_with_samples = {}

# Read the first N lines to check for column names and sample their contents
N = 100  # Adjust based on how many lines you want to sample
with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    for _ in range(N):
        line = file.readline()
        # Break if the end of file is reached
        if not line:
            break
        json_object = json.loads(line)
        for key, value in json_object.items():
            # If the column is seen for the first time, initialize its samples list
            if key not in columns_with_samples:
                columns_with_samples[key] = []
            # Add the value to the samples list if it's unique and the list is not full
            if value not in columns_with_samples[key] and len(columns_with_samples[key]) < 5:
                columns_with_samples[key].append(value)

# Convert the dictionary to a DataFrame for a more readable table
df_samples = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in columns_with_samples.items()]))
df_samples = df_samples.rename_axis('Sample #').reset_index()

print("Sample values for columns found in the first {} lines:".format(N))
print(df_samples.to_string(index=False))  # Use to_string for better formatting in console
