import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import calendar
import time
import config


def SplitData(X,Y, training_rate,valid_rate,seed=0):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    N, M = X.shape
    K = Y.shape[0]
    M_train = int(M * training_rate)
    M_valid = int(M * (valid_rate+training_rate))
    DATA = torch.cat((X, Y),dim=0)
    #index = np.arange(0, M, 1)
    #np.random.shuffle(index)
    index=torch.randperm(M)
    DATA = DATA[:, index]
    X = DATA[:-K, :].reshape(N, -1)
    Y = DATA[-K:, :].reshape(K, -1)
    X_train = X[:, :M_train]
    Y_train = Y[:, :M_train]
    X_valid = X[:, M_train:M_valid]
    Y_valid = Y[:, M_train:M_valid]
    X_test = X[:, M_valid:]
    Y_test = Y[:, M_valid:]
    return X_train, Y_train, X_valid, Y_valid,X_test,Y_test


def training_pNN(NN, train_loader, valid_loader, optimizer, lossfunction, Epoch=10**10):
    training_ID = ts = int(calendar.timegm(time.gmtime()))
    #print(f'The ID for this training is {training_ID}.')
    
    valid_loss = []
    valid_acc = []
    best_valid_loss = 100000
    not_decrease = 0
    
    for epoch in range(Epoch):
        for x_train, y_train in train_loader:
            prediction = NN(x_train)
            loss = lossfunction(prediction, y_train)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = NN(x_valid)
                loss_valid = lossfunction(prediction_valid, y_valid)
                
                yhat_valid = torch.argmax(prediction_valid.data, 2)
                
                yy_valid = y_valid.repeat(config.N, 1)
                valid_correct = torch.sum(yhat_valid == yy_valid.data)
                acc_valid = valid_correct / (y_valid.numel() * config.N)
                #valid_acc.append(acc_valid)

        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            torch.save(NN, f'./temp/NN_scheduler_temp_{training_ID}')
            best = acc_valid
            not_decrease = 0
        else:
            not_decrease += 1
        
        #valid_loss.append(loss_valid.data)
        
        if not_decrease > 5000:
            #print('Early stop.')
            
            break
            
        if not epoch % 5000:
            print(f'| Epoch: {epoch:-8d} | Valid accuracy: {acc_valid:.5f} | Valid loss: {loss_valid.data:.9f} |')
            
            
    #print('Finished.')
    return torch.load(f'./temp/NN_scheduler_temp_{training_ID}'), best
