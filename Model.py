import torch
import torch.nn as nn
import torch.nn.functional as F
from Params import args
import numpy as np


class Model(nn.Module):
    def __init__(self, emb_dim, content_dim, item_freq=None):
        super(Model, self).__init__()
        self.emb_dim = emb_dim
        self.content_dim = content_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.one_labels = torch.ones(args.batch_size, dtype=torch.float32)

        self.freq_coef_a = args.freq_coef_a
        self.freq_coef_M = args.freq_coef_M
        # scaled_item_freq = torch.tanh(self.freq_coef_a * item_freq)
        # Clipping the result
        # self.item_weight = torch.clamp(
        #     scaled_item_freq, min=0, max=torch.tanh(self.freq_coef_M)
        # )

        # f_U and f_I
        self.transformed_layers = [200, 200]

        # Define batch normalization layers
        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(self.transformed_layers[i])
                for i in range(len(self.transformed_layers) - 1)
            ]
        )

        # Define layers for item and user embeddings
        self.item_layers = nn.ModuleList()
        self.user_layers = nn.ModuleList()
        for i in range(len(self.transformed_layers) - 1):
            if i == 0:
                item_linear = nn.Linear(self.content_dim, self.transformed_layers[i])
                user_linear = nn.Linear(self.emb_dim, self.transformed_layers[i])
                # nn.init.normal_(item_linear.weight, std=0.01)
                # nn.init.normal_(user_linear.weight, std=0.01)
                self.item_layers.append(item_linear)
                self.user_layers.append(user_linear)
            else:
                item_linear = nn.Linear(
                    self.transformed_layers[i - 1], self.transformed_layers[i]
                )
                user_linear = nn.Linear(
                    self.transformed_layers[i - 1], self.transformed_layers[i]
                )
                # nn.init.normal_(item_linear.weight, std=0.01)
                # nn.init.normal_(user_linear.weight, std=0.01)
                self.item_layers.append(item_linear)
                self.user_layers.append(user_linear)

        # Define output layers
        self.item_output = nn.Linear(
            self.transformed_layers[0], self.transformed_layers[1]
        )
        self.user_output = nn.Linear(
            self.transformed_layers[0], self.transformed_layers[1]
        )
        # nn.init.normal_(self.item_output.weight, std=0.01)
        # nn.init.normal_(self.user_output.weight, std=0.01)

    def forward(self, user_emb, item_content_emb):
        gen_user_emb = user_emb
        # Forward pass through item and user layers
        for i, (il, ul, bn) in enumerate(
            zip(self.item_layers, self.user_layers, self.batch_norms)
        ):
            item_content_emb = torch.tanh(bn(il(item_content_emb.float())))
            gen_user_emb = torch.tanh(bn(ul(gen_user_emb)))
        # Output layer
        gen_item_emb = self.item_output(item_content_emb)
        gen_user_emb = self.user_output(gen_user_emb)

        return gen_user_emb, gen_item_emb

    def calcLoss(self, user_emb, item_content_emb, item_behavior_emb):
        gen_user_emb, gen_item_emb = self.forward(user_emb, item_content_emb)
        # Separate item content embeddings into positive and negative
        separate_item_content_emb = gen_item_emb.view(2, -1, self.emb_dim)
        pos_item_content_emb = separate_item_content_emb[0].squeeze(0)
        neg_item_content_emb = separate_item_content_emb[1].squeeze(0)

        # Separate item behavior embeddings into positive and negative
        separate_item_behavior_emb = item_behavior_emb.view(2, -1, self.emb_dim)
        pos_item_behavior_emb = separate_item_behavior_emb[0].squeeze(0)
        neg_item_behavior_emb = separate_item_behavior_emb[1].squeeze(0)

        # Joint distillation loss
        # Supervised loss
        student_pos_logit = torch.sum(gen_user_emb * pos_item_content_emb, dim=1)
        student_neg_logit = torch.sum(gen_user_emb * neg_item_content_emb, dim=1)
        student_rank_distance = student_pos_logit - student_neg_logit
        self.one_labels = torch.ones(
            student_rank_distance.shape[0], dtype=torch.float32
        )
        supervised_loss = F.binary_cross_entropy_with_logits(
            student_rank_distance, self.one_labels
        )

        teacher_pos_logit = torch.sum(user_emb * pos_item_behavior_emb, dim=1)
        teacher_neg_logit = torch.sum(user_emb * neg_item_behavior_emb, dim=1)
        teacher_rank_distance = teacher_pos_logit - teacher_neg_logit

        # rating distribution difference
        rate_diff = self.gamma * torch.mean(
            torch.abs(teacher_pos_logit - student_pos_logit)
            + torch.abs(teacher_neg_logit - student_neg_logit)
        )

        # Ranking difference
        # separate_item_freq = self.item_weight.view(2, -1)
        # pos_item_freq = separate_item_freq[0]
        pos_item_freq = 1

        rank_diff = self.alpha * torch.mean(
            pos_item_freq
            * F.binary_cross_entropy_with_logits(
                student_rank_distance, torch.sigmoid(teacher_rank_distance)
            )
        )

        # identification difference
        student_ii_logit = torch.sum(pos_item_content_emb * pos_item_content_emb, dim=1)
        student_ij_logit = torch.mean(
            torch.mm(pos_item_content_emb, neg_item_content_emb.t()), dim=1
        )
        student_iden_distance = student_ii_logit - student_ij_logit

        teacher_ii_logit = torch.sum(
            pos_item_behavior_emb * pos_item_behavior_emb, dim=1
        )
        teacher_ij_logit = torch.mean(
            torch.mm(pos_item_behavior_emb, neg_item_behavior_emb.t()), dim=1
        )
        teacher_iden_distance = teacher_ii_logit - teacher_ij_logit

        iden_diff = self.beta * torch.mean(
            pos_item_freq
            * F.binary_cross_entropy_with_logits(
                input=student_iden_distance, target=torch.sigmoid(teacher_iden_distance)
            )
        )

        distill_loss = rate_diff + rank_diff + iden_diff
        total_loss = supervised_loss + distill_loss
        return total_loss

    def get_user_rating(self, user_emb, item_content_emb, user_index, item_index):
        gen_user_emb, gen_item_emb = self.forward(
            user_emb[user_index], item_content_emb[item_index]
        )
        cold_prediction = gen_user_emb @ gen_item_emb.T
        return F.sigmoid(cold_prediction)

    def get_ranked_rating(self, ratings, k=20):
        ranked_scores, ranked_indices = torch.topk(ratings, k=k, dim=1)
        return ranked_scores, ranked_indices
