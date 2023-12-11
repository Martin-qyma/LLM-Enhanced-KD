import torch
import torch.nn as nn
import torch.nn.functional as F
from Params import args


def truncated_normal_(tensor, mean=0, std=0.01):
    with torch.no_grad():
        size = tensor.nelement()
        tmp = tensor.new_empty(size + size // 2).normal_(mean=mean, std=std)
        valid = (tmp < 2 * std) & (tmp > -2 * std)
        ind = valid.nonzero().view(-1)
        if ind.size(0) < size:
            raise ValueError("Truncated normal initialization failed")
        tensor.data.copy_(tmp[ind[:size]].view_as(tensor))


class BPRMF(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, reg_rate):
        super(BPRMF, self).__init__()
        self.user_emb = nn.Embedding(user_num, latent_dim)
        self.item_emb = nn.Embedding(item_num, latent_dim)
        self.reg_rate = reg_rate

        # Initialize factors
        # nn.init.normal_(self.user_emb.weight, std=0.01)
        # nn.init.normal_(self.item_emb.weight, std=0.01)
        truncated_normal_(self.user_emb.weight, std=0.01)
        truncated_normal_(self.item_emb.weight, std=0.01)

    def forward(self, u_index, pos_i_index, neg_i_index):
        user_embedding = self.user_emb(u_index)
        pos_item_emb = self.item_emb(pos_i_index)
        neg_item_emb = self.item_emb(neg_i_index)

        pos_pred = (user_embedding * pos_item_emb).sum(dim=1)
        neg_pred = (user_embedding * neg_item_emb).sum(dim=1)

        # Regularization loss
        reg_loss = (
            self.reg_rate
            * 1
            / 2
            * (
                torch.norm(user_embedding)
                + torch.norm(pos_item_emb)
                + torch.norm(neg_item_emb)
            )
        ) / args.batch_size

        return pos_pred, neg_pred, reg_loss

    def bpr_loss_function(self, pos_pred, neg_pred, reg_loss):
        self.one_labels = torch.ones(len(pos_pred), dtype=torch.float32)
        bpr_loss = torch.mean(
            F.binary_cross_entropy_with_logits(pos_pred - neg_pred, self.one_labels)
        )
        # bpr_loss = -torch.mean(F.logsigmoid(pos_pred - neg_pred))
        # Combine the BPR loss and the regularization term
        total_loss = bpr_loss + reg_loss
        return total_loss

    def get_user_rating(self, user, item):
        # Predict the score for user and item
        user_factors = self.user_emb(user)
        item_factors = self.item_emb(item)
        return F.sigmoid(user_factors @ item_factors.T)

    def get_ranked_rating(self, ratings, k=20):
        ranked_scores, ranked_indices = torch.topk(ratings, k=k, dim=1)
        return ranked_scores, ranked_indices
