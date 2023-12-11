import pandas as pd
import numpy as np
import pickle, json

# Load the final mappings
user_mapping_file_path = "./data/user_mapping.json"
with open(user_mapping_file_path, "r") as f:
    user_mapping = json.load(f)
item_mapping_file_path = "./data/item_mapping.json"
with open(item_mapping_file_path, "r") as f:
    item_mapping = json.load(f)

user_num = len(user_mapping)
item_num = len(item_mapping)

# Load the csv files
filename = "./data/distribution_shift/warm/warm_train.csv"
data = pd.read_csv(filename)
warm_train_user_nb = [[] for _ in range(user_num)]
warm_train_item_nb = [[] for _ in range(item_num)]
warm_train_item_nb_num = [0 for _ in range(len(data["item"].unique()))]
for index, row in data.iterrows():
    warm_train_user_nb[row["user"]].append(row["item"])
    warm_train_item_nb[row["item"]].append(row["user"])
    warm_train_item_nb_num[row["item"]] += 1

filename = "./data/distribution_shift/warm/warm_val.csv"
data = pd.read_csv(filename)
warm_val_user_nb = [[] for _ in range(user_num)]
warm_val_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    warm_val_user_nb[row["user"]].append(row["item"])
    warm_val_item_nb[row["item"]].append(row["user"])

filename = "./data/distribution_shift/warm/warm_test.csv"
data = pd.read_csv(filename)
warm_test_user_nb = [[] for _ in range(user_num)]
warm_test_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    warm_test_user_nb[row["user"]].append(row["item"])
    warm_test_item_nb[row["item"]].append(row["user"])

filename = "./data/distribution_shift/cold/cold_val.csv"
data = pd.read_csv(filename)
cold_val_user_nb = [[] for _ in range(user_num)]
cold_val_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    cold_val_user_nb[row["user"]].append(row["item"])
    cold_val_item_nb[row["item"]].append(row["user"])

filename = "./data/distribution_shift/cold/cold_test.csv"
data = pd.read_csv(filename)
cold_test_user_nb = [[] for _ in range(user_num)]
cold_test_item_nb = [[] for _ in range(item_num)]
for index, row in data.iterrows():
    cold_test_user_nb[row["user"]].append(row["item"])
    cold_test_item_nb[row["item"]].append(row["user"])

# Convert dim1 list to numpy array
# warm_train_user_nb = [np.array(sublist) for sublist in warm_train_user_nb]
# warm_train_item_nb = [np.array(sublist) for sublist in warm_train_item_nb]
# warm_val_user_nb = [np.array(sublist) for sublist in warm_val_user_nb]
# warm_val_item_nb = [np.array(sublist) for sublist in warm_val_item_nb]
# warm_test_user_nb = [np.array(sublist) for sublist in warm_test_user_nb]
# warm_test_item_nb = [np.array(sublist) for sublist in warm_test_item_nb]
# cold_val_user_nb = [np.array(sublist) for sublist in cold_val_user_nb]
# cold_val_item_nb = [np.array(sublist) for sublist in cold_val_item_nb]
# cold_test_user_nb = [np.array(sublist) for sublist in cold_test_user_nb]
# cold_test_item_nb = [np.array(sublist) for sublist in cold_test_item_nb]

para_dict = {
    "warm_train_user_nb": warm_train_user_nb,
    "warm_train_item_nb": warm_train_item_nb,
    "warm_train_item_nb_num": warm_train_item_nb_num,
    "warm_val_user_nb": warm_val_user_nb,
    "warm_val_item_nb": warm_val_item_nb,
    "warm_test_user_nb": warm_test_user_nb,
    "warm_test_item_nb": warm_test_item_nb,
    "cold_val_user_nb": cold_val_user_nb,
    "cold_val_item_nb": cold_val_item_nb,
    "cold_test_user_nb": cold_test_user_nb,
    "cold_test_item_nb": cold_test_item_nb,
    "user_num": user_num,
    "item_num": item_num,
}

with open("./data/para_dict.pickle", "wb") as handle:
    pickle.dump(para_dict, handle)
print("para_dict saved")
