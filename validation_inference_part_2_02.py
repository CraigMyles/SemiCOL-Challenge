import pandas as pd
import json
import warnings

csv_input_path = "./class_counts/all_class_counts.csv"

# Read the CSV data
data = pd.read_csv(csv_input_path)

# Extract the filename from the input path
filename = data["filename"][0]

# Convert data types to numeric
data.iloc[:, 2:] = data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

# Calculate row sums
row_sums = data.iloc[:, 2:].sum(axis=1)

# Convert counts to percentages
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    data.iloc[:, 2:] = data.iloc[:, 2:].div(row_sums, axis=0) * 100

# Add filename column to output
data.insert(0, "filename", data.pop("filename"))

# Save the result as a new CSV
data.to_csv(csv_input_path.replace(".csv", "_percentages.csv"), index=False)

# Initialize an empty dictionary for storing the results
result = {}

# Iterate through the DataFrame
for index, row in data.iterrows():
    filename = row['filename']
    class_1_value = row['class_1']
    class_3_value = row['class_3']
    class_7_value = row['class_7']
    
    # Check the conditions and store the result in the dictionary
    if class_1_value > 3.5 and class_3_value > 0.85 and class_7_value > 0.07:
        result[filename] = 1.0
    else:
        result[filename] = 0.0

# Save the dictionary as a JSON file
with open('/predict/classification.json', 'w') as outfile:
    json.dump(result, outfile, indent=4)
