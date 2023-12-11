import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset

class BPRData(Dataset):
    def __init__(self, user_item_pairs, num_items, negative_samples=1):
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, item = self.user_item_pairs[idx]
        neg_items = [np.random.choice(self.num_items) for _ in range(self.negative_samples)]
        return {
            'user': torch.tensor(user, dtype=torch.long),
            'pos_item': torch.tensor(item, dtype=torch.long),
            'neg_items': torch.tensor(neg_items, dtype=torch.long),
        }

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, factor_num, reg_rate):
        super(BPRMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, factor_num)
        self.item_embedding = nn.Embedding(num_items, factor_num)
        self.reg_rate = reg_rate

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        return (user_emb * item_emb).sum(1)

    def bpr_loss(self, user, pos_item, neg_items):
        pos_pred = self.forward(user, pos_item)
        neg_pred = self.forward(user, neg_items)
        loss = -torch.log(torch.sigmoid(pos_pred - neg_pred)).mean()
        reg_loss = self.reg_rate * 0.5 * (user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2))
        return loss + reg_loss

def train(model, data_loader, optimizer, epochs):
    for epoch in range(epochs):
        for idx, data in enumerate(data_loader):
            user = data['user']
            pos_item = data['pos_item']
            neg_items = data['neg_items']

            optimizer.zero_grad()
            loss = model.bpr_loss(user, pos_item, neg_items)
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{idx}/{len(data_loader)}], Loss: {loss.item()}')

# Assuming num_users and num_items are the number of users and items in your dataset
# and factor_num is the number of latent factors you want to have.
# user_item_pairs is a list of tuples (user_id, item_id) indicating the interactions.
# You would need to load your dataset and create this list.

num_users, num_items, factor_num, reg_rate = 1000, 1000, 64, 1e-3  # Replace these with your actual values
user_item_pairs = []  # Replace with your actual data

dataset = BPRData(user_item_pairs=user_item_pairs, num_items=num_items)
data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

model = BPRMF(num_users, num_items, factor_num, reg_rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, data_loader, optimizer, epochs=5)
