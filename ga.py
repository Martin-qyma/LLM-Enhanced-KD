import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Constants
NUM_USERS = 1000  # Number of users
NUM_ITEMS = 1000  # Number of items
NUM_FACTORS = 64  # Number of latent factors for the embeddings
NUM_INTERACTIONS = 10000  # Number of interactions to simulate

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulate some data
user_ids = np.random.randint(NUM_USERS, size=NUM_INTERACTIONS)
item_ids = np.random.randint(NUM_ITEMS, size=NUM_INTERACTIONS)
ratings = np.ones(NUM_INTERACTIONS)  # Implicit feedback (1 for interactions)

# Create a set for faster look up
interactions_set = set(zip(user_ids, item_ids))


# Dataset
class BPRDataset(Dataset):
    def __init__(self, num_users, num_items, interactions_set):
        self.num_users = num_users
        self.num_items = num_items
        self.interactions_set = interactions_set
        self.total_interactions = list(interactions_set)

    def __len__(self):
        return len(self.interactions_set)

    def __getitem__(self, idx):
        # Positive sample
        user, positive_item = self.total_interactions[idx]

        # Negative sample (sample until we find an item the user hasn't interacted with)
        negative_item = np.random.randint(self.num_items)
        while (user, negative_item) in self.interactions_set:
            negative_item = np.random.randint(self.num_items)

        return user, positive_item, negative_item


# Create the dataset and data loader
dataset = BPRDataset(NUM_USERS, NUM_ITEMS, interactions_set)
data_loader = DataLoader(dataset, batch_size=256, shuffle=True)  # Shuffle for training

# Checking if the data loader works
for batch_idx, (user, positive_item, negative_item) in enumerate(data_loader):
    if batch_idx > 1:  # Just to check the first couple of batches
        break


class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, factors=NUM_FACTORS):
        super(BPRMF, self).__init__()
        self.user_factors = nn.Embedding(num_users, factors)
        self.item_factors = nn.Embedding(num_items, factors)

        # Initialize factors
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        # Get the latent factors for users and items
        user_factors = self.user_factors(user)
        item_i_factors = self.item_factors(item_i)
        item_j_factors = self.item_factors(item_j)

        # Predict the preference
        prediction_i = (user_factors * item_i_factors).sum(dim=1)
        prediction_j = (user_factors * item_j_factors).sum(dim=1)

        return (
            prediction_i - prediction_j
        )  # We return the difference for the pairwise ranking loss

    def predict(self, user, item):
        # Predict the score for a user and an item
        user_factors = self.user_factors(user)
        item_factors = self.item_factors(item)
        return (user_factors * item_factors).sum(dim=1)


# Loss function
def bpr_loss(prediction_diff):
    return -torch.log(torch.sigmoid(prediction_diff)).mean()


# Initialize the BPR-MF model
model = BPRMF(NUM_USERS, NUM_ITEMS, NUM_FACTORS)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the number of epochs for training
NUM_EPOCHS = 5

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0

    for batch_idx, (user, positive_item, negative_item) in enumerate(data_loader):
        # Convert data to PyTorch tensors and move it to the current device (CPU in this case)
        user = user.long()
        positive_item = positive_item.long()
        negative_item = negative_item.long()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute the prediction difference
        prediction_diff = model(user, positive_item, negative_item)

        # Compute loss
        loss = bpr_loss(prediction_diff)

        # Backward pass: Compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Accumulate epoch loss
        epoch_loss += loss.item()

    # Print the epoch loss
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(data_loader)}")


def create_validation_data(num_users, num_items, interactions_set, num_val=1000):
    """
    Create a validation set by generating a list of items for each user,
    ensuring one relevant item is included in the list.
    """
    val_data = []
    for _ in range(num_val):
        # Randomly select a user and a positive item from the interactions set
        user, positive_item = list(interactions_set)[
            np.random.randint(len(interactions_set))
        ]
        # Add to the validation set
        val_data.append((user, positive_item))
    return val_data


# Create a synthetic validation set
validation_data = create_validation_data(
    NUM_USERS, NUM_ITEMS, interactions_set, num_val=1000
)


# Function to calculate NDCG for a single user
def ndcg_at_k(ranked_list, pos_item, k=10):
    """
    Calculate NDCG@k for a single list of ranked items.
    """
    # Find the index of the positive item in the ranked list
    pos_index = np.where(ranked_list == pos_item)[0][0]
    # Calculate DCG@k
    dcg_at_k = 1.0 / np.log2(
        pos_index + 2
    )  # We add 2 because the index is 0-based and log2(1) is 0
    # Calculate IDCG@k (ideal DCG, the best possible DCG)
    idcg_at_k = (
        1.0  # The best possible DCG is when the positive item is at the first place
    )
    # Calculate NDCG@k
    ndcg = dcg_at_k / idcg_at_k
    return ndcg


# Function to evaluate the model on the validation set
def evaluate_model(model, validation_data, k=10):
    """
    Evaluate the model on the validation set using NDCG@k.
    """
    model.eval()  # Set the model to evaluation mode
    ndcg_scores = []
    with torch.no_grad():
        for user, positive_item in validation_data:
            # Predict the scores for all items for this user
            user_tensor = torch.tensor([user] * NUM_ITEMS, dtype=torch.long)
            items_tensor = torch.tensor(list(range(NUM_ITEMS)), dtype=torch.long)
            predictions = model.predict(user_tensor, items_tensor).cpu().numpy()

            # Rank items according to the predictions
            ranked_items = np.argsort(-predictions)  # Negative for descending order

            # Calculate NDCG for the ranked list
            ndcg_score = ndcg_at_k(ranked_items, positive_item, k)
            ndcg_scores.append(ndcg_score)

    # Calculate the average NDCG@k over all users in the validation set
    mean_ndcg = np.mean(ndcg_scores)
    return mean_ndcg


# Evaluate the trained model
ndcg_score = evaluate_model(model, validation_data, k=10)
print(ndcg_score)
