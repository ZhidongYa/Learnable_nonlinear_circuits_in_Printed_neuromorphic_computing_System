#!/usr/bin/env python

#SBATCH --job-name=LNC_RT

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hzhao@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1

import sys
sys.path.append("/pfs/data5/home/kit/tm/px3192/useful_code/LNC/")

import pNN_NetLevel_LNC_variation_circuit as pNN
import os
import math
import torch
import random
import pickle
import calendar
import numpy as np
import config
import training 
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.set_default_tensor_type(torch.FloatTensor)

seed = int(sys.argv[1])
epsilon = float(sys.argv[2])
strategy = int(sys.argv[3])

datasets = os.listdir('./dataset')
datasets = [f for f in datasets if (f.startswith('Dataset') and f.endswith('.p'))]
datasets.sort()

for dataset in datasets:

    datapath = os.path.join(f'./dataset/{dataset}')
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
    X_train    = data['X_train']
    y_train    = data['y_train']
    X_valid    = data['X_valid']
    y_valid    = data['y_valid']
    X_test     = data['X_test']
    y_test     = data['y_test']
    data_name  = data['name']

    N_class    = data['n_class']
    N_feature  = data['n_feature']
    N_train    = X_train.shape[0]
    N_valid    = X_valid.shape[0]
    N_test     = X_test.shape[0]

    print(f'Dataset "{data_name}" has {N_feature} input features and {N_class} classes.\nThere are {N_train} training examples, {N_valid} valid examples, and {N_test} test examples in the dataset.')

    # generate tensordataset
    trainset = TensorDataset(X_train, y_train)
    validset = TensorDataset(X_valid, y_valid)
    testset  = TensorDataset(X_test, y_test)

    # batch
    train_loader = DataLoader(trainset, batch_size=N_train)
    valid_loader = DataLoader(validset, batch_size=N_valid)
    test_loader  = DataLoader(testset,  batch_size=N_test)

    lr_theta = 0.1
    
    for lr_rt in [1., 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.]:
        setup = f'dataset:{data_name}_epsilon:{epsilon}_strategy:{strategy}_lr_rt:{lr_rt}_seed:{seed}'
        print(setup)

        if os.path.exists(f'./results_variation_circuit/pNN_{setup}'):
            print('File exists, pass.')
        else:
            config.SetSeed(seed)
            model = pNN.pNN([N_feature, 3, N_class], config.N, epsilon)
            
            lossfunction = pNN.lossfunction
            
            if strategy == 0:
                optimizer = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':lr_theta},
                                              {'params':model.GetParam('rt'), 'lr':lr_rt}])
                model, train_loss, valid_loss = training.train_nn(model, train_loader, valid_loader, lossfunction, optimizer, UUID=setup)
            
            elif strategy == 1:
                optimizer1 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':lr_theta},
                                              {'params':model.GetParam('rt'), 'lr':0}])
                optimizer2 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':0},
                                              {'params':model.GetParam('rt'), 'lr':lr_rt}])
                optimizer3 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':0},
                                              {'params':model.GetParam('rt'), 'lr':0}])
                model, train_loss, valid_loss = training.training_alternative(model, lossfunction, train_loader, valid_loader,
                                                                              optimizer1, optimizer2, optimizer3,
                                                                              UUID=setup)
            torch.save(model, f'./results_variation_circuit/pNN_{setup}')