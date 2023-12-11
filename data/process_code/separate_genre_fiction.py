import pandas as pd
from sklearn.model_selection import train_test_split
import networkx as nx
import csv


def k_core(mapped_data, k=10):
    G = nx.Graph()

    # Add edges to the graph from interactions
    for interaction in mapped_data:
        user = interaction["user"]
        item = interaction["item"]
        G.add_edge(user, item)

    # Find k-core of the graph
    k_core_graph = nx.k_core(G, k)

    # Extract interactions that are in k-core
    k_core_interactions = []
    users = []
    items = []
    for interaction in mapped_data:
        user = interaction["user"]
        item = interaction["item"]

        if user in k_core_graph and item in k_core_graph:
            k_core_interactions.append({"user": user, "item": item})
            users.append(user)
            items.append(item)

    return k_core_interactions, users, items


# Load the data
filename = "./data/distribution_shift/warm/genre_fiction.csv"
data = pd.read_csv(filename)
mapped_data = []
for index, row in data.iterrows():
    mapped_data.append({"user": row["user"], "item": row["item"]})
mapped_data, users, items = k_core(mapped_data, k=10)

# Define the output CSV file path
output_file_path = "./data/distribution_shift/warm/warm_dense.csv"

# Write the mapped data to the CSV file
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

data = pd.read_csv(output_file_path)

# Separate the data into training, validation, and test sets with a ratio of 8:1:1
train_data, temp_data = train_test_split(
    data, test_size=0.2, random_state=42
)  # 80% training, 20% temporary

users_trained = []
for index, row in train_data.iterrows():
    users_trained.append(row["user"])
for index, row in temp_data.iterrows():
    if row["user"] not in users_trained:
        temp_data.drop(index, inplace=True)

# split temporary data into 50% validation and 50% test
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# If you want to save the split data to separate CSV files:
train_data.to_csv("./data/distribution_shift/warm/gf_train.csv", index=False)
validation_data.to_csv("./data/distribution_shift/warm/gf_val.csv", index=False)
test_data.to_csv("./data/distribution_shift/warm/gf_test.csv", index=False)
