import json, csv, os
from tqdm import tqdm
from collections import defaultdict
import networkx as nx


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
    for interaction in mapped_data:
        user = interaction["user"]
        item = interaction["item"]

        if user in k_core_graph and item in k_core_graph:
            k_core_interactions.append({"user": user, "item": item})

    return k_core_interactions


file_path = "../source/review_warm.json"

num_lines = sum(1 for line in open(file_path, 'r'))
data = []
with open(file_path, "r") as file:
    for line in tqdm(file, total=num_lines, desc="Parsing data", unit=" lines"):
        # Parse each line as a JSON object and append to the data list
        process = json.loads(line)
        dict = {}
        dict["reviewerID"] = process["reviewerID"]
        dict["asin"] = process["asin"]
        dict["overall"] = process["overall"]
        data.append(dict)

# Dictionaries to store the mappings of reviewerID and asin to unique numbers
reviewer_id_mapping = defaultdict(lambda: len(reviewer_id_mapping) + 1)

asin_mapping_file_path = "../asin_mapping.json"
with open(asin_mapping_file_path, "r") as f:
    # Load existing asin_mapping from a file and convert it to defaultdict
    asin_mapping = json.load(f)

# Filter the data to include only those entries where "overall" is larger or equal to 3.0 and in the asin_mapping
filtered_data = [
    entry
    for entry in data
    if (entry.get("overall", 0) > 3.0) and (entry.get("asin", None) in asin_mapping)
]

# Apply the mappings to the filtered data
mapped_data = []
for entry in filtered_data:
    reviewer_id = entry.get("reviewerID", None)
    asin = entry.get("asin", None)
    if reviewer_id and asin:  # Ensure both IDs are present
        mapped_reviewer_id = reviewer_id_mapping[reviewer_id]
        mapped_asin = asin_mapping[asin]
        mapped_data.append({"user": mapped_reviewer_id, "item": mapped_asin})

# Display the first entry of the mapped data and the total number of unique reviewerID and asin
print(
    mapped_data[0] if mapped_data else {}, len(reviewer_id_mapping), len(asin_mapping)
)

mapped_data = k_core(mapped_data, k=10)
print(len(mapped_data))
# Define the output CSV file path
output_file_path = "../total.csv"

# Write the mapped data to the CSV file
with open(output_file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["user", "item"])
    writer.writeheader()
    writer.writerows(mapped_data)

print(f"Processing completed. Mapped item content is saved to {output_file_path}")

reviewer_id_mapping_file_path = "../reviewerID_mapping.json"
with open(reviewer_id_mapping_file_path, "w") as f:
    json.dump(reviewer_id_mapping, f)

print(
    f"Processing completed. Reviewer_id_mapping is saved to {reviewer_id_mapping_file_path}"
)
