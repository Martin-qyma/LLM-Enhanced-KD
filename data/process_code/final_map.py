import pandas as pd
from collections import defaultdict
import json

user_mapping = defaultdict(lambda: len(user_mapping))
item_mapping = defaultdict(lambda: len(item_mapping))
# Load the data
filename = "./data/distribution_shift/warm/warm_dense.csv"
data = pd.read_csv(filename)
for index, row in data.iterrows():
    mapped_user = user_mapping[str(row["user"])]
    mapped_item = item_mapping[str(row["item"])]
filename = "./data/distribution_shift/cold/cold_dense.csv"
data = pd.read_csv(filename)
for index, row in data.iterrows():
    mapped_user = user_mapping[str(row["user"])]
    mapped_item = item_mapping[str(row["item"])]

filepath = "./data/user_mapping.json"
with open(filepath, "w") as f:
    json.dump(user_mapping, f)
filepath = "./data/item_mapping.json"
with open(filepath, "w") as f:
    json.dump(item_mapping, f)
print("Mappings saved")
