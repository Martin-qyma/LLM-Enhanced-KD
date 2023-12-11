import json
import pandas as pd
import numpy as np

# Load the final item mapping
item_mapping_file_path = "./data/item_mapping.json"
with open(item_mapping_file_path, "r") as f:
    item_mapping = json.load(f)
used_items = item_mapping.keys()

# Load the BERT encodings
BERT_file_path = "./data/BERT_encodings.json"
with open(BERT_file_path, "r") as f:
    BERT_encodings = json.load(f)

item_content_emb = np.zeros((len(item_mapping), 768))
for key in BERT_encodings.keys():
    if key in used_items:
        item_content_emb[item_mapping[str(key)]] = BERT_encodings[key]

# Save the embedding as numpy array
np.save("./data/item_content_emb.npy", item_content_emb)
print("item_content_emb saved")
