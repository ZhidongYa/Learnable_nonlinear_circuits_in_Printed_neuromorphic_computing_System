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
    K = Y.shape[1]
    M_train = int(N * training_rate)
    M_valid = int(N * (valid_rate + training_rate))
    DATA = torch.cat((X, Y),dim = 1)
    index=torch.randperm(N)
    DATA = DATA[index, :]
    X = DATA[:, :-K].reshape(N, -1)
    Y = DATA[:, -K:].reshape(N, -1)
    X_train = X[:M_train, :]
    Y_train = Y[:M_train, :]
    X_valid = X[M_train:M_valid, :]
    Y_valid = Y[M_train:M_valid, :]
    X_test = X[M_valid:, :]
    Y_test = Y[M_valid:, :]
    return X_train, Y_train, X_valid, Y_valid,X_test,Y_test


class Layer(torch.nn.Module):
    def __init__(self,in_d,out_d):
        super().__init__()
        self.layer=torch.nn.Linear(in_d,out_d)
        
    
    def forward(self, x):         
        x=self.layer(x)
        return x
    
class DynamicNet(torch.nn.Module):
    def __init__(self, topology):
        super().__init__()
        self.model = torch.nn.Sequential()
        for l in range(len(topology)-2):
            self.model.add_module(f'Layer {l}', Layer(topology[l], topology[l+1]))
            self.model.add_module(f'activation {l}' , torch.nn.PReLU())
        self.model.add_module(f'Layer {l+1}',Layer(topology[-2], topology[-1]))
      
    def forward(self, x):                      
        return self.model(x)
    
def training_pNN(NN, lossfunction, train_loader, valid_loader, optimizer,  Epoch=10**10):
    training_ID = ts = int(calendar.timegm(time.gmtime()))
    print(f'The ID for this training is {training_ID}.')
    
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

        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            torch.save(NN, f'./inv_temp/NN_scheduler_temp_{training_ID}')
            not_decrease = 0
        else:
            not_decrease += 1
        
        #valid_loss.append(loss_valid.data)
        
        if not_decrease > 500:
            print('Early stop.')
            break
            
        if not epoch % 500:
            print(f'| Epoch: {epoch:-8d} | Valid loss: {loss_valid.data:.9f} |')
            
            
    print('Finished.')
    return torch.load(f'./inv_temp/NN_scheduler_temp_{training_ID}'), valid_loss, valid_acc