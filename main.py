import os
import torch
import numpy as np
import pandas as pd
from Model import Model
import argparse, Metrics
from Params import args
from DataHandler import DataHandler
from BPRMF import BPRMF
from MetaModel import MetaModel
import pickle, json, random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)


class ALDI:
    def __init__(self, handler):
        self.handler = handler
        args.dataset = "CiteULIke"

    def train_teacher(self):
        with open("./data/CiteULike/convert_dict.pkl", "rb") as file:
            para_dict = pickle.load(file)
        args.data = "warm"
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        model = BPRMF(
            user_num=user_num,
            item_num=item_num,
            latent_dim=200,
            reg_rate=0.001,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_value = 0.0
        for epoch in range(1000):
            model.train()
            total_loss = 0.0
            for i, trnbatch in enumerate(trnLoader):
                users, pos_items, neg_items = trnbatch
                # Zero the gradients
                optimizer.zero_grad()
                outputs = model.forward(
                    u_index=users,
                    pos_i_index=pos_items,
                    neg_i_index=neg_items,
                )
                pos_pred, neg_pred, reg_loss = outputs
                loss = model.bpr_loss_function(
                    pos_pred=pos_pred, neg_pred=neg_pred, reg_loss=reg_loss
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate the model and tell when to stop
            model.eval()
            with torch.no_grad():
                rating_list = []
                score_list = []
                groundTrue_list = []
                for valbatch in valLoader:
                    users, items = valbatch
                    predictions = model.get_user_rating(
                        users, torch.tensor(list(range(item_num)))
                    )
                    ranked_scores, ranked_indices = model.get_ranked_rating(
                        predictions, k=20
                    )
                    groundTrue = para_dict["warm_val_user_nb"][users]
                    score_list.append(ranked_scores.tolist())
                    rating_list.append(ranked_indices.tolist())
                    groundTrue_list.append(groundTrue.tolist())

                X = zip(rating_list, groundTrue_list)
                pre_results = list(map(Metrics.test_one_batch, X))
                results = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                }
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
                n_ts_user = float(len(np.unique(valLoader.dataset.users)))
                results["recall"] /= n_ts_user
                results["precision"] /= n_ts_user
                results["ndcg"] /= n_ts_user

                val_value = results["recall"].item()
                if val_value > best_val_value:
                    patient_count = 0
                    best_val_value = val_value
                    # Save the model and embeddings
                    user_embeddings = model.user_emb.weight.detach()
                    item_embeddings = model.item_emb.weight.detach()
                    with open("./data/CiteULike/user_emb.pkl", "wb") as file:
                        pickle.dump(user_embeddings, file)
                    with open("./data/CiteULike/item_behavior_emb.pkl", "wb") as file:
                        pickle.dump(item_embeddings, file)
                    torch.save(model.state_dict(), "./data/CiteULike/BPRMF.pkl")

                print(
                    f"Epoch {epoch}, Loss: {total_loss/len(trnbatch):.4f}, Patience: {patient_count}, Recall: {val_value:.4f}"
                )
                if patient_count >= args.patience:
                    break
                patient_count += 1

    def test_teacher(self):
        with open("./data/CiteULike/convert_dict.pkl", "rb") as file:
            para_dict = pickle.load(file)
        args.data = "warm"
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        unique_users = len(np.unique(testLoader.dataset.users))
        print("Data Loaded")

        model = BPRMF(
            user_num=user_num,
            item_num=item_num,
            latent_dim=200,
            reg_rate=0.001,
        )
        model.load_state_dict(torch.load("./data/CiteULike/BPRMF.pkl"))
        model.eval()

        with torch.no_grad():
            rating_list = []
            score_list = []
            groundTrue_list = []
            for testbatch in testLoader:
                users, items = testbatch
                predictions = model.get_user_rating(
                    users, torch.tensor(list(range(item_num)))
                )
                ranked_scores, ranked_indices = model.get_ranked_rating(
                    predictions, k=20
                )
                groundTrue = para_dict["warm_test_user_nb"][users]
                score_list.append(ranked_scores.tolist())
                rating_list.append(ranked_indices.tolist())
                groundTrue_list.append(groundTrue.tolist())

            X = zip(rating_list, groundTrue_list)
            pre_results = list(map(Metrics.test_one_batch, X))
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
            }
            for result in pre_results:
                results["recall"] += result["recall"]
                results["precision"] += result["precision"]
                results["ndcg"] += result["ndcg"]
            n_ts_user = float(unique_users)
            results["recall"] /= n_ts_user
            results["precision"] /= n_ts_user
            results["ndcg"] /= n_ts_user
            print(
                f"Recall: {results['recall'].item():.4};  NDCG: {results['ndcg'].item():.4}"
            )
            return results, np.concatenate(score_list, axis=0)

    def train_student(self):
        with open("./data/CiteULike/convert_dict.pkl", "rb") as file:
            para_dict = pickle.load(file)
        args.data = "warm"
        trnLoader, _, _, _ = DataHandler().LoadData()
        args.data = "cold"
        _, valLoader, _, _ = DataHandler().LoadData()
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        item_content_emb = torch.tensor(
            np.load("./data/CiteULike/CiteULike_item_content.npy")
        )
        with open("./data/CiteULike/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/CiteULike/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        print("Data loaded")

        # calculate item frequency
        item_freq = np.ones(shape=(para_dict["item_num"],), dtype=np.float32)
        item_to_user_neighbors = para_dict["emb_item_nb"][para_dict["warm_item"]]
        for item_index, user_neighbor_list in zip(
            para_dict["warm_item"], item_to_user_neighbors
        ):
            item_to_item_neighborhoods = para_dict["emb_user_nb"][user_neighbor_list]
            item_freq[item_index] = sum(
                [1.0 / len(neighborhood) for neighborhood in item_to_item_neighborhoods]
            )
        x_expect = (len(trnLoader.dataset.users) / para_dict["item_num"]) * (
            1 / (len(trnLoader.dataset.users) / para_dict["user_num"])
        )
        args.freq_coef_a = args.freq_coef_M / x_expect

        args.freq_coef_M = torch.tensor(args.freq_coef_M, dtype=torch.float32)
        args.freq_coef_a = torch.tensor(args.freq_coef_a, dtype=torch.float32)

        model = Model(emb_dim=200, content_dim=300, item_freq=torch.tensor(item_freq))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        best_val_value = 0.0
        for epoch in range(1000):
            model.train()
            total_loss = 0.0
            for trnbatch in trnLoader:
                users, pos_items, neg_items = trnbatch
                items = torch.cat((pos_items, neg_items), dim=-1)
                # Zero the gradients
                optimizer.zero_grad()
                loss = model.calcLoss(
                    user_emb=user_emb[users],
                    item_content_emb=item_content_emb[items],
                    item_behavior_emb=item_behavior_emb[items],
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                rating_list = []
                score_list = []
                groundTrue_list = []
                for valbatch in valLoader:
                    users, items = valbatch
                    predictions = model.get_user_rating(
                        user_emb=user_emb,
                        item_content_emb=item_content_emb,
                        user_index=users,
                        item_index=torch.tensor(list(range(item_num))),
                    )
                    ranked_scores, ranked_indices = model.get_ranked_rating(
                        predictions, k=20
                    )
                    groundTrue = para_dict["cold_val_user_nb"][users]
                    score_list.append(ranked_scores.tolist())
                    rating_list.append(ranked_indices.tolist())
                    groundTrue_list.append(groundTrue.tolist())

                X = zip(rating_list, groundTrue_list)
                pre_results = list(map(Metrics.test_one_batch, X))
                results = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                }
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
                n_ts_user = float(len(np.unique(valLoader.dataset.users)))
                results["recall"] /= n_ts_user
                results["precision"] /= n_ts_user
                results["ndcg"] /= n_ts_user

                val_value = results["recall"].item()
                if val_value > best_val_value:
                    patient_count = 0
                    best_val_value = val_value
                    # Save the model
                    torch.save(model.state_dict(), "./data/CiteULike/student.pkl")

                print(
                    f"Epoch {epoch}, Loss: {total_loss/len(trnbatch):.4f}, Patience: {patient_count}, Recall: {val_value:.4f}"
                )
                if patient_count >= args.patience:
                    break
                patient_count += 1

    def test_student(self):
        args.data = "cold"
        with open("./data/CiteULike/convert_dict.pkl", "rb") as file:
            para_dict = pickle.load(file)
        item_content_emb = torch.tensor(
            np.load("./data/CiteULike/CiteULike_item_content.npy")
        )
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        # Get user and item embeddings
        with open("./data/CiteULike/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/CiteULike/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        # with open("./data/BERT_encodings.json", "r") as file:
        #     content_data = json.load(file)

        args.freq_coef_M = torch.tensor(args.freq_coef_M, dtype=torch.float32)
        args.freq_coef_a = torch.tensor(args.freq_coef_a, dtype=torch.float32)
        model = Model(emb_dim=200, content_dim=300, item_freq=torch.tensor([]))
        model.load_state_dict(torch.load("./data/CiteULike/student.pkl"))
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        unique_users = len(np.unique(testLoader.dataset.users))
        print("Data Loaded")
        model.eval()

        with torch.no_grad():
            rating_list = []
            score_list = []
            groundTrue_list = []
            for testbatch in testLoader:
                users, items = testbatch
                predictions = model.get_user_rating(
                    user_emb=user_emb,
                    item_content_emb=item_content_emb,
                    user_index=users,
                    item_index=torch.tensor(list(range(item_num))),
                )
                ranked_scores, ranked_indices = model.get_ranked_rating(
                    predictions, k=20
                )
                groundTrue = para_dict["cold_test_user_nb"][users]
                score_list.append(ranked_scores.tolist())
                rating_list.append(ranked_indices.tolist())
                groundTrue_list.append(groundTrue.tolist())

            X = zip(rating_list, groundTrue_list)
            pre_results = list(map(Metrics.test_one_batch, X))
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
            }
            for result in pre_results:
                results["recall"] += result["recall"]
                results["precision"] += result["precision"]
                results["ndcg"] += result["ndcg"]
            n_ts_user = float(unique_users)
            results["recall"] /= n_ts_user
            results["precision"] /= n_ts_user
            results["ndcg"] /= n_ts_user
            print(
                f"Recall: {results['recall'].item():.4};  NDCG: {results['ndcg'].item():.4}"
            )
            return results, np.concatenate(score_list, axis=0)


class Recommender:
    def __init__(self, handler):
        args.dataset = "Amazon2018"
        self.handler = handler

    def train_teacher(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        args.data = "warm"
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        print("Data Loaded")

        model = BPRMF(
            user_num=user_num,
            item_num=item_num,
            latent_dim=200,
            reg_rate=0.001,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_value = 0.0
        for epoch in range(1000):
            model.train()
            total_loss = 0.0
            for i, trnbatch in enumerate(trnLoader):
                users, pos_items, neg_items = trnbatch
                # Zero the gradients
                optimizer.zero_grad()
                outputs = model.forward(
                    u_index=users,
                    pos_i_index=pos_items,
                    neg_i_index=neg_items,
                )
                pos_pred, neg_pred, reg_loss = outputs
                loss = model.bpr_loss_function(
                    pos_pred=pos_pred, neg_pred=neg_pred, reg_loss=reg_loss
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate the model and tell when to stop
            model.eval()
            with torch.no_grad():
                rating_list = []
                score_list = []
                groundTrue_list = []
                for valbatch in valLoader:
                    users, items = valbatch
                    predictions = model.get_user_rating(
                        users, torch.tensor(list(range(item_num)))
                    )
                    ranked_scores, ranked_indices = model.get_ranked_rating(
                        predictions, k=20
                    )
                    groundTrue = [para_dict["warm_val_user_nb"][i] for i in users]
                    score_list.append(ranked_scores.tolist())
                    rating_list.append(ranked_indices.tolist())
                    groundTrue_list.append(groundTrue)

                X = zip(rating_list, groundTrue_list)
                pre_results = list(map(Metrics.test_one_batch, X))
                results = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                }
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
                n_ts_user = float(len(np.unique(valLoader.dataset.users)))
                results["recall"] /= n_ts_user
                results["precision"] /= n_ts_user
                results["ndcg"] /= n_ts_user

                val_value = results["recall"].item()
                if val_value > best_val_value:
                    patient_count = 0
                    best_val_value = val_value
                    # Save the model and embeddings
                    user_embeddings = model.user_emb.weight.detach()
                    item_embeddings = model.item_emb.weight.detach()
                    with open("data/Amazon2018/user_emb.pkl", "wb") as file:
                        pickle.dump(user_embeddings, file)
                    with open("data/Amazon2018/item_behavior_emb.pkl", "wb") as file:
                        pickle.dump(item_embeddings, file)
                    torch.save(model.state_dict(), "./data/Amazon2018/BPRMF.pkl")

                print(
                    f"Epoch {epoch}, Loss: {total_loss/len(trnbatch):.4f}, Patience: {patient_count}, Recall: {val_value:.4f}"
                )
                if patient_count >= args.patience:
                    break
                patient_count += 1

    def test_teacher(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        args.data = "warm"
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        unique_users = len(np.unique(testLoader.dataset.users))
        print("Data Loaded")

        model = BPRMF(
            user_num=user_num,
            item_num=item_num,
            latent_dim=200,
            reg_rate=0.001,
        )
        model.load_state_dict(torch.load("./data/Amazon2018/BPRMF.pkl"))
        model.eval()

        with torch.no_grad():
            rating_list = []
            score_list = []
            groundTrue_list = []
            for testbatch in testLoader:
                users, items = testbatch
                predictions = model.get_user_rating(
                    users, torch.tensor(list(range(item_num)))
                )
                ranked_scores, ranked_indices = model.get_ranked_rating(
                    predictions, k=20
                )
                groundTrue = [para_dict["warm_test_user_nb"][i] for i in users]
                score_list.append(ranked_scores.tolist())
                rating_list.append(ranked_indices.tolist())
                groundTrue_list.append(groundTrue)

            X = zip(rating_list, groundTrue_list)
            pre_results = list(map(Metrics.test_one_batch, X))
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
            }
            for result in pre_results:
                results["recall"] += result["recall"]
                results["precision"] += result["precision"]
                results["ndcg"] += result["ndcg"]
            n_ts_user = float(unique_users)
            results["recall"] /= n_ts_user
            results["precision"] /= n_ts_user
            results["ndcg"] /= n_ts_user
            print(
                f"Recall: {results['recall'].item():.4};  NDCG: {results['ndcg'].item():.4}"
            )
            return results, np.concatenate(score_list, axis=0)

    def train_student(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        args.data = "warm"
        trnLoader, _, _, _ = DataHandler().LoadData()
        args.data = "cold"
        _, valLoader, _, _ = DataHandler().LoadData()
        with open("./data/Amazon2018/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/Amazon2018/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        item_content_emb = torch.tensor(np.load("./data/item_content_emb.npy"))
        print("Data loaded")

        # # calculate item frequency
        # item_freq = np.ones(shape=(para_dict["item_num"],), dtype=np.float32)
        # item_to_user_neighbors = para_dict["emb_item_nb"][para_dict["warm_item"]]
        # for item_index, user_neighbor_list in zip(
        #     para_dict["warm_item"], item_to_user_neighbors
        # ):
        #     item_to_item_neighborhoods = para_dict["emb_user_nb"][user_neighbor_list]
        #     item_freq[item_index] = sum(
        #         [1.0 / len(neighborhood) for neighborhood in item_to_item_neighborhoods]
        #     )
        # x_expect = (len(trnLoader.dataset.users) / para_dict["item_num"]) * (
        #     1 / (len(trnLoader.dataset.users) / para_dict["user_num"])
        # )
        # args.freq_coef_a = args.freq_coef_M / x_expect

        # args.freq_coef_M = torch.tensor(args.freq_coef_M, dtype=torch.float32)
        # args.freq_coef_a = torch.tensor(args.freq_coef_a, dtype=torch.float32)

        model = Model(emb_dim=200, content_dim=768)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        best_val_value = 0.0
        for epoch in range(1000):
            model.train()
            total_loss = 0.0
            for trnbatch in trnLoader:
                users, pos_items, neg_items = trnbatch
                items = torch.cat((pos_items, neg_items), dim=-1)
                # Zero the gradients
                optimizer.zero_grad()
                loss = model.calcLoss(
                    user_emb=user_emb[users],
                    item_content_emb=item_content_emb[items],
                    item_behavior_emb=item_behavior_emb[items],
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                rating_list = []
                score_list = []
                groundTrue_list = []
                for valbatch in valLoader:
                    users, items = valbatch
                    predictions = model.get_user_rating(
                        user_emb=user_emb,
                        item_content_emb=item_content_emb,
                        user_index=users,
                        item_index=torch.tensor(list(range(item_num))),
                    )
                    ranked_scores, ranked_indices = model.get_ranked_rating(
                        predictions, k=20
                    )
                    groundTrue = [para_dict["cold_val_user_nb"][i] for i in users]
                    score_list.append(ranked_scores.tolist())
                    rating_list.append(ranked_indices.tolist())
                    groundTrue_list.append(groundTrue)

                X = zip(rating_list, groundTrue_list)
                pre_results = list(map(Metrics.test_one_batch, X))
                results = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                }
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
                n_ts_user = float(len(np.unique(valLoader.dataset.users)))
                results["recall"] /= n_ts_user
                results["precision"] /= n_ts_user
                results["ndcg"] /= n_ts_user

                val_value = results["recall"].item()
                if val_value > best_val_value:
                    patient_count = 0
                    best_val_value = val_value
                    # Save the model
                    torch.save(model.state_dict(), "./data/Amazon2018/student.pkl")

                print(
                    f"Epoch {epoch}, Loss: {total_loss/len(trnbatch):.4f}, Patience: {patient_count}, Recall: {val_value:.4f}"
                )
                if patient_count >= args.patience:
                    break
                patient_count += 1

    def test_student(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        args.data = "cold"
        trnLoader, valLoader, testLoader, _ = DataHandler().LoadData()
        with open("./data/Amazon2018/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/Amazon2018/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        item_content_emb = torch.tensor(np.load("./data/item_content_emb.npy"))
        print("Data loaded")

        # args.freq_coef_M = torch.tensor(args.freq_coef_M, dtype=torch.float32)
        # args.freq_coef_a = torch.tensor(args.freq_coef_a, dtype=torch.float32)
        model = Model(emb_dim=200, content_dim=768)
        model.load_state_dict(torch.load("./data/Amazon2018/student.pkl"))
        unique_users = len(np.unique(testLoader.dataset.users))
        model.eval()

        with torch.no_grad():
            rating_list = []
            score_list = []
            groundTrue_list = []
            for testbatch in testLoader:
                users, items = testbatch
                predictions = model.get_user_rating(
                    user_emb=user_emb,
                    item_content_emb=item_content_emb,
                    user_index=users,
                    item_index=torch.tensor(list(range(item_num))),
                )
                ranked_scores, ranked_indices = model.get_ranked_rating(
                    predictions, k=20
                )
                groundTrue = [para_dict["cold_test_user_nb"][i] for i in users]
                score_list.append(ranked_scores.tolist())
                rating_list.append(ranked_indices.tolist())
                groundTrue_list.append(groundTrue)

            X = zip(rating_list, groundTrue_list)
            pre_results = list(map(Metrics.test_one_batch, X))
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
            }
            for result in pre_results:
                results["recall"] += result["recall"]
                results["precision"] += result["precision"]
                results["ndcg"] += result["ndcg"]
            n_ts_user = float(unique_users)
            results["recall"] /= n_ts_user
            results["precision"] /= n_ts_user
            results["ndcg"] /= n_ts_user
            print(
                f"Recall: {results['recall'].item():.4};  NDCG: {results['ndcg'].item():.4}"
            )
            return results, np.concatenate(score_list, axis=0)

    def train_meta(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        args.data = "warm"
        trnLoader, _, _, _ = DataHandler().LoadData()
        args.data = "cold"
        _, valLoader, _, _ = DataHandler().LoadData()
        with open("./data/Amazon2018/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/Amazon2018/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        item_content_emb = torch.tensor(np.load("./data/item_content_emb.npy"))
        print("Data loaded")

        model = MetaModel(emb_dim=200, content_dim=768)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        best_val_value = 0.0
        for epoch in range(1000):
            model.train()
            total_loss = 0.0
            for trnbatch in trnLoader:
                users, pos_items, neg_items = trnbatch
                items = torch.cat((pos_items, neg_items), dim=-1)
                # Zero the gradients
                optimizer.zero_grad()
                loss = model.calcLoss(
                    user_emb=user_emb[users],
                    item_content_emb=item_content_emb[items],
                    item_behavior_emb=item_behavior_emb[items],
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                rating_list = []
                score_list = []
                groundTrue_list = []
                for valbatch in valLoader:
                    users, items = valbatch
                    predictions = model.get_user_rating(
                        user_emb=user_emb,
                        item_content_emb=item_content_emb,
                        user_index=users,
                        item_index=torch.tensor(list(range(item_num))),
                    )
                    ranked_scores, ranked_indices = model.get_ranked_rating(
                        predictions, k=20
                    )
                    groundTrue = [para_dict["cold_val_user_nb"][i] for i in users]
                    score_list.append(ranked_scores.tolist())
                    rating_list.append(ranked_indices.tolist())
                    groundTrue_list.append(groundTrue)

                X = zip(rating_list, groundTrue_list)
                pre_results = list(map(Metrics.test_one_batch, X))
                results = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "ndcg": 0.0,
                }
                for result in pre_results:
                    results["recall"] += result["recall"]
                    results["precision"] += result["precision"]
                    results["ndcg"] += result["ndcg"]
                n_ts_user = float(len(np.unique(valLoader.dataset.users)))
                results["recall"] /= n_ts_user
                results["precision"] /= n_ts_user
                results["ndcg"] /= n_ts_user

                val_value = results["recall"].item()
                if val_value > best_val_value:
                    patient_count = 0
                    best_val_value = val_value
                    # Save the model
                    torch.save(model.state_dict(), "./data/Amazon2018/meta.pkl")

                print(
                    f"Epoch {epoch}, Loss: {total_loss/len(trnbatch):.4f}, Patience: {patient_count}, Recall: {val_value:.4f}"
                )
                if patient_count >= args.patience:
                    break
                patient_count += 1

    def test_meta(self):
        with open("./data/Amazon2018/para_dict.pickle", "rb") as file:
            para_dict = pickle.load(file)
        user_num = para_dict["user_num"]
        item_num = para_dict["item_num"]
        args.data = "cold"
        _, _, testLoader, _ = DataHandler().LoadData()
        with open("./data/Amazon2018/user_emb.pkl", "rb") as file:
            user_emb = pickle.load(file)
        with open("./data/Amazon2018/item_behavior_emb.pkl", "rb") as file:
            item_behavior_emb = pickle.load(file)
        item_content_emb = torch.tensor(np.load("./data/item_content_emb.npy"))
        print("Data loaded")

        model = MetaModel(emb_dim=200, content_dim=768)
        model.load_state_dict(torch.load("./data/Amazon2018/meta.pkl"), strict=False)
        unique_users = len(np.unique(testLoader.dataset.users))
        model.eval()

        with torch.no_grad():
            rating_list = []
            score_list = []
            groundTrue_list = []
            for testbatch in testLoader:
                users, items = testbatch
                predictions = model.get_user_rating(
                    user_emb=user_emb,
                    item_content_emb=item_content_emb,
                    user_index=users,
                    item_index=torch.tensor(list(range(item_num))),
                )
                ranked_scores, ranked_indices = model.get_ranked_rating(
                    predictions, k=20
                )
                groundTrue = [para_dict["cold_test_user_nb"][i] for i in users]
                score_list.append(ranked_scores.tolist())
                rating_list.append(ranked_indices.tolist())
                groundTrue_list.append(groundTrue)

            X = zip(rating_list, groundTrue_list)
            pre_results = list(map(Metrics.test_one_batch, X))
            results = {
                "precision": 0.0,
                "recall": 0.0,
                "ndcg": 0.0,
            }
            for result in pre_results:
                results["recall"] += result["recall"]
                results["precision"] += result["precision"]
                results["ndcg"] += result["ndcg"]
            n_ts_user = float(unique_users)
            results["recall"] /= n_ts_user
            results["precision"] /= n_ts_user
            results["ndcg"] /= n_ts_user
            print(
                f"Recall: {results['recall'].item():.4};  NDCG: {results['ndcg'].item():.4}"
            )
            return results, np.concatenate(score_list, axis=0)


if __name__ == "__main__":
    # ALDI = ALDI(DataHandler())
    # ALDI.train_teacher()
    # ALDI.test_teacher()
    # ALDI.train_student()
    # ALDI.test_student()

    Recommender = Recommender(DataHandler)
    # Recommender.train_teacher()
    # Recommender.test_teacher()
    # Recommender.train_student()
    # Recommender.test_student()
    Recommender.train_meta()
    Recommender.test_meta()
