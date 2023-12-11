import numpy as np
import torch
import json
import pickle

with open("./data/CiteULike/convert_dict.pkl", "rb") as file:
    para_dict = pickle.load(file)
print(para_dict.keys())
