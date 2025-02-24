# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:32:08 2022

@author: shefai
"""
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
import time
import networkx as nx
import pickle
from algorithms.GRU4Rec_PyTorch_Official.gru4rec_pytorch import SessionDataIterator
from algorithms.GRU4Rec_PyTorch_Official.gru4rec_pytorch import *


import warnings
warnings.filterwarnings("ignore")
import os

def init_seed(seed):
    np.random.seed(seed)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


class GRURecPytorch:
    def __init__(self, epoch = 1, loss = 'cross-entropy', constrained_embedding = True, embedding = 100, elu_param= 1, layers = 512, batch_size = 300, 
                 dropout_p_embed =  0.5, dropout_p_hidden = 0.3, learning_rate = 0.05, momentum = 0.4, n_sample = 2048, sample_alpha = 0.2, bpreg = 0.9, logq=0,  seed = 2000, device = 'cuda:0'):
           
        self.seed = seed
        init_seed(self.seed)
        self.epoch = epoch
        self.constrained_embedding = constrained_embedding
        self.embedding = embedding
        self.elu_param = elu_param
        self.batch_size = batch_size
        self.layers = [layers]
        self.dropout_p_embed = dropout_p_embed
        self.sample_store_size = 10000000
        self.loss = loss
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.bpreg = bpreg
        self.logq = logq
        self.sessionid = -1
        self.device = device
        
        self.session_key = "SessionId"
        self.item_key = 'ItemId'
        self.time_key = 'Time'
        
        
    def fit(self, train, test):
        print("Model training is started")
        init_seed(self.seed)
        self.gru = GRU4Rec(layers=self.layers, loss=self.loss, batch_size=self.batch_size, dropout_p_embed=self.dropout_p_embed,
                 dropout_p_hidden=self.dropout_p_hidden, learning_rate=self.learning_rate, momentum=self.momentum, sample_alpha=self.sample_alpha, n_sample=self.n_sample, embedding=self.embedding,
                 constrained_embedding=self.constrained_embedding, n_epochs=self.epoch, bpreg=self.bpreg, elu_param=self.elu_param, logq=self.logq, device=self.device)
        
        self.gru.fit(train, sample_cache_max_size=self.sample_store_size, item_key=self.item_key, session_key = self.session_key, time_key=self.time_key)

        encodingToitemID = {v: k for k, v in self.gru.data_iterator.itemidmap.to_dict().items()}

        self.encodingToitemID = pd.Series(encodingToitemID)
        
        #res = self.batch_eval(self.gru, test, batch_size=1, cutoff= [20], item_key=self.item_key, session_key=self.session_key, time_key=self.time_key)
        print("Model training is completed")
                 
    def predict_next(self, sid, prev_iid, items_to_predict, timestamp):
        
        init_seed(self.seed)
        columns = [self.session_key, self.item_key, self.time_key]

        if(sid !=self.sessionid):
            # Create an empty DataFrame with these columns
            self.testList = pd.DataFrame(columns=columns)
            self.sessionid = sid
        tempDF = pd.DataFrame(columns=columns)
        tempDF[self.session_key] = [sid]
        tempDF[self.item_key] = [prev_iid]
        tempDF[self.time_key] = [timestamp]
  
        self.testList = pd.concat([self.testList, tempDF])
        #if self.testList.shape[0] == 1:
            #self.testList = pd.concat([self.testList, tempDF])
        
        predition = self.batch_eval(self.gru, self.testList, batch_size=1, cutoff= 100, item_key=self.item_key, session_key=self.session_key, time_key=self.time_key)
        return predition
   
    def clear(self):
        pass


    @torch.no_grad()   
    def batch_eval(self, gru, test_data, cutoff=100, batch_size=1, item_key='ItemId', session_key='SessionId', time_key='Time'):
        if gru.error_during_train: 
            raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
        H = []
        for i in range(len(gru.layers)):
            H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
        reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
        data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)
        for in_idxs in data_iterator.test_data(enable_neg_samples=False, reset_hook=reset_hook):
            for h in H: h.detach_()
            O = gru.model.forward(in_idxs, H, None, training=False)
            oscores = O.T
            oscores = oscores.flatten().cpu()
            score, index = torch.topk(oscores, cutoff)
            index = [self.encodingToitemID[i]  for i in index.numpy()]
            predition = pd.Series(score.numpy(), index = index)
            predition.sort_values(ascending=False, inplace=True)
        return predition
    

    """
    @torch.no_grad()   
    def batch_eval(self, gru, test_data, cutoff=[20], batch_size=1, item_key='ItemId', session_key='SessionId', time_key='Time'):
        if gru.error_during_train: 
            raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
        H = []
        for i in range(len(gru.layers)):
            H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
        reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
        data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, device=gru.device, itemidmap=gru.data_iterator.itemidmap)
        hit = 0
        sum = 0
        for in_idxs, out_idxs in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
            for h in H: h.detach_()
            O = gru.model.forward(in_idxs, H, None, training=False)
            oscores = O.T
            if len(out_idxs) > 1:
                print(len(out_idxs))
            
            oscores = oscores.flatten().cpu()
            score, index = torch.topk(oscores, 100)
            predition = pd.Series(score.numpy(), index = index.numpy())
            predition.sort_values(ascending=False, inplace=True)
            target_value = out_idxs.flatten().cpu().numpy()[0]
            if target_value in predition[:20]:
                hit = hit + 1

            sum = sum + 1
            
        print(hit / sum)
        return hit / sum

        """



















        