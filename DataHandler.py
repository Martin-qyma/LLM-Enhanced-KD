import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import random, copy
from Params import args


class DataHandler:
    def __init__(self):
        if args.dataset == "Amazon2018":
            predir = "./data/distribution_shift/"
            trainfile_path = predir + "warm/warm_train.csv"
            if args.data == "warm":
                valfile_path = predir + "warm/warm_val.csv"
                testfile_path = predir + "warm/warm_test.csv"
            else:
                valfile_path = predir + "cold/cold_val.csv"
                testfile_path = predir + "cold/cold_test.csv"
            self.trnfile_path = trainfile_path
            self.valfile_path = valfile_path
            self.tstfile_path = testfile_path
        else:
            self.trnfile_path = "./data/CiteULike/warm_emb.csv"
            if args.data == "warm":
                self.valfile_path = "./data/CiteULike/warm_val.csv"
                self.tstfile_path = "./data/CiteULike/warm_test.csv"
            else:
                self.valfile_path = "./data/CiteULike/cold_item_val.csv"
                self.tstfile_path = "./data/CiteULike/cold_item_test.csv"

    def LoadCSV(self, file_path):
        with open(file_path, mode="r") as file:
            csvFile = csv.reader(file)
            # Skipping the first row
            next(csvFile, None)
            users_id = []
            items_id = []
            u_p = {}
            for lines in csvFile:
                # users_id.append(int(lines[0]) - 1)
                # items_id.append(int(lines[1]) - 100000001)
                # u_p[int(lines[0]) - 1] = int(lines[1]) - 100000001
                users_id.append(int(lines[0]))
                items_id.append(int(lines[1]))
                u_p[int(lines[0])] = int(lines[1])
        return users_id, items_id, u_p

    def negSampling(self):
        users_id, items_id, u_p = self.LoadCSV(self.trnfile_path)
        neg_items = []
        for user in users_id:
            neg_item = random.choice(items_id)
            while neg_item == u_p[user]:
                tem_list = copy.deepcopy(items_id)
                tem_list.remove(u_p[user])
                neg_item = random.choice(tem_list)
            neg_items.append(neg_item)
        trn_batch = np.stack((users_id, items_id, neg_items), axis=0)
        trn_batch = torch.from_numpy(trn_batch)
        return trn_batch

    def LoadData(self):
        trn_batch = self.negSampling()
        trnData = TrnData(trn_batch)
        trnLoader = DataLoader(trnData, batch_size=args.batch_size, shuffle=True)

        users_id, items_id, u_p = self.LoadCSV(self.tstfile_path)
        temp_data = np.stack((users_id, items_id), axis=0)
        tst_batch = torch.from_numpy(temp_data)
        tstData = TstData(tst_batch)
        tstLoader = DataLoader(tstData, batch_size=args.batch_size, shuffle=True)

        users_id, items_id, u_p = self.LoadCSV(self.valfile_path)
        temp_data = np.stack((users_id, items_id), axis=0)
        val_batch = torch.from_numpy(temp_data)
        valData = TstData(val_batch)
        valLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=True)

        return trnLoader, valLoader, tstLoader, u_p


class TrnData(Dataset):
    def __init__(self, trn_batch):
        self.trn_batch = trn_batch
        self.users = trn_batch[0]
        self.pos_items = trn_batch[1]
        self.neg_items = trn_batch[2]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index], self.pos_items[index], self.neg_items[index]


class TstData(Dataset):
    def __init__(self, tst_batch):
        self.tst_batch = tst_batch
        self.users = tst_batch[0]
        self.pos_items = tst_batch[1]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index], self.pos_items[index]
