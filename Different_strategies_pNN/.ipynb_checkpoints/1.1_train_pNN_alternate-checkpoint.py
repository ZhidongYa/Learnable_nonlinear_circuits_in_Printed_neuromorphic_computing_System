#!/usr/bin/env python

#SBATCH --job-name=LNC_Alternative

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

learning_strategy = int(sys.argv[1]) # 0 for neuron level, 1 for layer level, 2 for net level
training_strategy = int(sys.argv[4]) # 0 for simutanous, 1 for alternative

if learning_strategy == 0:
    import pNN_NeuronLevel_LNC as pNN
elif learning_strategy == 1:
    import pNN_LayerLevel_LNC as pNN
elif learning_strategy == 2:
    import pNN_NetLevel_LNC as pNN
else:
    pass

# packages
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

    lr_theta = float(sys.argv[2])
    lr_lnc = float(sys.argv[3])

    for seed in range(10):
        config.SetSeed(seed)

        setup = f'dataset:{data_name}_learning_strategy:{learning_strategy}_training_strategy:{training_strategy}_lr_theta:{lr_theta}_lr_lnc:{lr_lnc}_seed:{seed}'
        print(setup)

        if os.path.exists(f'./results_different_strategy/pNN_{setup}'):
            print('File exists, pass.')
        else:
            
            model = pNN.pNN([N_feature, 3, N_class])
            
            lossfunction = pNN.lossfunction
            
            if training_strategy == 1:
                optimizer1 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':lr_theta},
                                              {'params':model.GetParam('eta'), 'lr':0},
                                              {'params':model.GetParam('inv'),'lr':0}])
                optimizer2 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':0},
                                              {'params':model.GetParam('eta'), 'lr':lr_lnc},
                                              {'params':model.GetParam('inv'),'lr':0}])
                optimizer3 = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':0},
                                              {'params':model.GetParam('eta'), 'lr':0},
                                              {'params':model.GetParam('inv'),'lr':lr_lnc}])



                model, train_loss, valid_loss = training.training_alternative(model, lossfunction, train_loader, valid_loader,
                                                                                optimizer1, optimizer2, optimizer3,
                                                                                UUID=setup)
            elif training_strategy == 0:
                optimizer = torch.optim.Adam([{'params':model.GetParam('theta'),'lr':lr_theta},
                                              {'params':model.GetParam('eta'), 'lr':lr_lnc},
                                              {'params':model.GetParam('inv'),'lr':lr_lnc}])
                
                model, valid_loss, valid_acc_NN = training.train_nn(model, train_loader, valid_loader, lossfunction, optimizer, UUID=setup)
                
            torch.save(model, f'./results_different_strategy/pNN_{setup}')