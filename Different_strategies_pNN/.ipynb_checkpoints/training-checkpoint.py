import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import calendar
import time
import config
import math
import os

def train_nn(nn, train_loader, valid_loader, lossfunction, optimizer, UUID='default'):
    
    training_ID = int(calendar.timegm(time.gmtime()))
    if not UUID == 'default':
        UUID = f'{hash(UUID)}'
    print(f'The ID for this training is {UUID}_{training_ID}.')
    
    train_loss = []
    valid_loss = []
    best_valid_loss = math.inf
    patience = 0

    for epoch in range(10**10):
        for x_train, y_train in train_loader:
            prediction_train = nn(x_train)
            L_train = lossfunction(prediction_train, y_train)
            train_loss.append(L_train.item())
            
            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()
            
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                L_valid = lossfunction(prediction_valid, y_valid)
                valid_loss.append(L_valid.item())

        if L_valid.item() < best_valid_loss:
            best_valid_loss = L_valid.item()
            torch.save(nn, f'./temp/NN_{UUID}_{training_ID}')
            patience = 0
        else:
            patience += 1

        if patience > 5000:
            print('Early stop.')
            break

        if not epoch % 100:
            print(f'| Epoch: {epoch:-8d} | Train loss: {L_train.item():.5f} | Valid loss: {L_valid.item():.5f} |')
    
    # remove temp files
    resulted_nn = torch.load(f'./temp/NN_{UUID}_{training_ID}')
    os.remove(f'./temp/NN_{UUID}_{training_ID}')
    
    print('Finished.')
    return resulted_nn, train_loss, valid_loss


def training_alternative(nn, lossfunction, train_loader, valid_loader, optimizer1, optimizer2, optimizer3, UUID='default', N_alternate=1):
    training_ID = int(calendar.timegm(time.gmtime()))
    if not UUID == 'default':
        UUID = f'{hash(UUID)}'
    print(f'The ID for this training is {UUID}_{training_ID}.')
    
    train_loss = []
    valid_loss = []
    best_valid_loss = 10**10
    patience = 0
    
    for epoch in range(10**10):
        for x_train, y_train in train_loader:
            prediction = nn(x_train)
            loss = lossfunction(prediction, y_train)
            train_loss.append(loss.item())
            
            if epoch % (N_alternate*3) < N_alternate:
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
            elif epoch % (N_alternate*3) < N_alternate*2:
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
            else:
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()
            
        
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                loss_valid = lossfunction(prediction_valid, y_valid)
                valid_loss.append(loss_valid.item())

        if loss_valid.data < best_valid_loss:
            best_valid_loss = loss_valid.data
            torch.save(nn, f'./temp/NN_alternate_{UUID}_{training_ID}')
            patience = 0
        else:
            patience += 1
        
        if patience > 5000:
            print('Early stop.')
            break
            
        if not epoch % 100:
            print(f'| Epoch: {epoch:-8d} | Train loss: {loss.item():.5f} | Valid loss: {loss_valid.item():.5f} |')
            
    print('Finished.')
    
    # remove temp files
    resulted_nn = torch.load(f'./temp/NN_alternate_{UUID}_{training_ID}')
    os.remove(f'./temp/NN_alternate_{UUID}_{training_ID}')
    
    return resulted_nn, train_loss, valid_loss
