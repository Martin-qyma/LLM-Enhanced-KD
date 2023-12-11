import json
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

# Load JSON data
file_path = "../used_item.json"
data = []
with open(file_path, "r") as f:
    for line in f:
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize a dictionary to hold the encodings
encodings = {}

# Set the batch size
batch_size = 128


# Function to process and encode batches
def process_batch(batch):
    texts = []
    ids = []
    for id, details in batch:
        title = details.get("title", "")
        description = " ".join(details.get("description", []))
        text = title + ": " + description
        texts.append(text)
        ids.append(id)

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Compute the mean of the last hidden states and convert them to a list
    encodings_batch = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    return dict(zip(ids, encodings_batch))


# Flatten the nested dictionary and divide it into batches
items = []
for obj in data:
    items.extend(list(obj.items()))

num_batches = len(items) // batch_size + int(len(items) % batch_size > 0)
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch = items[start_idx:end_idx]
    encodings.update(process_batch(batch))

# Save encodings to a file
output_file_path = "../BERT_encodings.json"

# Save encodings to a JSON file
with open(output_file_path, "w") as f:
    json.dump(encodings, f)
