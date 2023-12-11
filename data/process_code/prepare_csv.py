import json
import pandas as pd
import csv

# Load the final mappings
user_mapping_file_path = "./data/user_mapping.json"
with open(user_mapping_file_path, "r") as f:
    user_mapping = json.load(f)
item_mapping_file_path = "./data/item_mapping.json"
with open(item_mapping_file_path, "r") as f:
    item_mapping = json.load(f)

# Load the csv files
filename = "./data/distribution_shift/warm/gf_train.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append(
        {"user": user_mapping[str(row["user"])], "item": item_mapping[str(row["item"])]}
    )
# Write new file
output_file_path = "./data/distribution_shift/warm/warm_train.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

# Load the csv files
filename = "./data/distribution_shift/warm/gf_val.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append(
        {"user": user_mapping[str(row["user"])], "item": item_mapping[str(row["item"])]}
    )
# Write new file
output_file_path = "./data/distribution_shift/warm/warm_val.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

# Load the csv files
filename = "./data/distribution_shift/warm/gf_test.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append(
        {"user": user_mapping[str(row["user"])], "item": item_mapping[str(row["item"])]}
    )
# Write new file
output_file_path = "./data/distribution_shift/warm/warm_test.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

# Load the csv files
filename = "./data/distribution_shift/cold/history_val.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append(
        {"user": user_mapping[str(row["user"])], "item": item_mapping[str(row["item"])]}
    )
# Write new file
output_file_path = "./data/distribution_shift/cold/cold_val.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

# Load the csv files
filename = "./data/distribution_shift/cold/history_test.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append(
        {"user": user_mapping[str(row["user"])], "item": item_mapping[str(row["item"])]}
    )
# Write new file
output_file_path = "./data/distribution_shift/cold/cold_test.csv"
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)
print("Final mapping successful")
