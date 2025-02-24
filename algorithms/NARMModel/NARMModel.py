#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""

import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn
from algorithms.NARMModel.utils import collate_fn
from algorithms.NARMModel.narm import *
from algorithms.NARMModel.dataset import load_data, RecSysDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_seed(seed = 2024):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

class NARMModel:
    def __init__(self, epoch = 10,  lr = 0.0001, batch_size = 100, embedding_size = 50, l2 = 0.00005, seed = 2000):

        self.batch_size = batch_size
        self.hidden_size = 100
        self.embed_dim = embedding_size
        self.epoch = epoch
        self.lr = lr
        self.lr_dc = 0.1
        self.lr_dc_step = 80
        self.seed = seed
        self.sessionid = -1

    def fit(self, train, test):
        init_seed(self.seed)
        session_key = "SessionId"
        item_key = "ItemId"
        CatId = "CatId"
        index_session = train.columns.get_loc( session_key)
        index_item = train.columns.get_loc( item_key )
        index_cat = train.columns.get_loc( CatId )
    
        session_item_train = {}
        # Convert the session data into sequence
        for row in train.itertuples(index=False):
            if row[index_session] in session_item_train:
                session_item_train[row[index_session]] += [(row[index_item])] 
            else: 
                session_item_train[row[index_session]] = [(row[index_item])]
        
        word2index ={}
        index2word = {}
        item_no = 0
        for key, values in session_item_train.items():
            length = len(session_item_train[key])
            for i in range(length):
                if session_item_train[key][i] in word2index:
                    session_item_train[key][i] = word2index[session_item_train[key][i]]
                else:
                    word2index[session_item_train[key][i]] = item_no
                    index2word[item_no] = session_item_train[key][i]
                    session_item_train[key][i] = item_no
                    item_no += 1
        features = []
        targets = []
        for value in session_item_train.values():
            for i in range(1, len(value)):
                targets += [value[-i]]
                features += [value[:-i]]
        session_item_test = {}
        # Convert the session data into sequence
        for row in test.itertuples(index=False):
            if row[index_session] in session_item_test:
                session_item_test[row[index_session]] += [(row[index_item])] 
            else: 
                session_item_test[row[index_session]] = [(row[index_item])]
                
        for key, values in session_item_test.items():
            length = len(session_item_test[key])
            for i in range(length):
                if session_item_test[key][i] in word2index:
                    session_item_test[key][i] = word2index[session_item_test[key][i]]
                else:
                    word2index[session_item_test[key][i]] = item_no
                    index2word[item_no] = session_item_test[key][i]
                    session_item_test[key][i] = item_no
                    item_no +=1


        features1 = []
        targets1 = []
        for value in session_item_test.values():
            for i in range(1, len(value)):
                targets1 += [value[-i]]
                features1 += [value[:-i]]
        self.num_node =  item_no
        self.word2index = word2index
        self.index2word = index2word 
        train_data = (features, targets)

        train_loader = RecSysDataset(train_data)
        train_loader = DataLoader(train_loader, batch_size = self.batch_size, shuffle = True, collate_fn = collate_fn)
        model = NARM(self.num_node, self.hidden_size, self.embed_dim, self.batch_size).to(device)

        optimizer = optim.Adam(model.parameters(), self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size = self.lr_dc_step, gamma = self.lr_dc)
        
        model.train()
        for epoch in tqdm(range(self.epoch)):
            scheduler.step(epoch = epoch)
            sum_epoch_loss = 0
            for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
                seq = seq.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                outputs = model(seq, lens)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step() 
                loss_val = loss.item()
                sum_epoch_loss += loss_val
            print(sum_epoch_loss)
        self.model = model
        
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        if(sid !=self.sessionid):
            self.testList = []
            self.sessionid = sid
        # incomming elements of a session....
        prev_iid = self.word2index[prev_iid]
        self.testList.append(prev_iid)
        # temp_list
        temp_list = []
        temp_list = ([self.testList], [prev_iid])
        
        valid_loader = RecSysDataset(temp_list) 
        valid_loader = DataLoader(valid_loader, batch_size = 1, shuffle = True, collate_fn = collate_fn)


        self.model.eval()
        with torch.no_grad():
            for seq, target, lens in valid_loader:
                seq = seq.to(device)
                target = target.to(device)
                outputs = self.model(seq, lens)
                logits = F.softmax(outputs, dim = 1)
                target = target.detach().cpu().numpy()
                
                sub_scores_k100_index = logits.topk(100)[1]
                sub_scores_k100_index = sub_scores_k100_index.detach().cpu().numpy()
                sub_scores_k100_index = np.ravel(sub_scores_k100_index)

                sub_scores_k100_score = logits.topk(100)[0]
                sub_scores_k100_score = sub_scores_k100_score.detach().cpu().numpy()
                sub_scores_k100_score = np.ravel(sub_scores_k100_score)
                
        tempList = []
        
        for key in sub_scores_k100_index:
            tempList.append(self.index2word[key])
        preds = pd.Series(data = list(sub_scores_k100_score), index = tempList)
        return preds
        

    def clear(self):
        pass
    
    
