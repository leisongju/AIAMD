import torch
import time
import torch.utils.data as Data
import os
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from IPython import display
import sys
import json
from sklearn   import   preprocessing
import pickle
import math
import copy
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def exp(labels):
    for i in range(len(labels)):
        for j in range(0,2):
            if labels[i,j] > 0:
                labels[i,j] = math.log(labels[i,j])
            else:
                labels[i,j] = -math.log(-labels[i,j])
    return labels

def inverse_exp(labels):
    for j in range(0,2):
        if labels[0,j] < 0:
            labels[0,j] = math.exp(labels[0,j])
        else:
            labels[0,j] = -math.exp(-labels[0,j])
    return labels


def e10(labels):
    for i in range(len(labels)):
        for j in range(0,2):
            if labels[i,j] > 0:
                labels[i,j] = math.log10(labels[i,j])
            else:
                labels[i,j] = -math.log10(-labels[i,j])
    return labels

def inverse_e10(labels):
    for j in range(0,2):
        if labels[0,j] < 0:
            labels[0,j] = math.pow(10,labels[0,j])
        else:
            labels[0,j] = -math.pow(10,-labels[0,j])
    return labels

def DataLoader(trainname, testname, num_input, num_output, batch_size, output_style=str,data_preprocessing_name=str):
    dt = pd.read_csv(trainname)
    train_features=dt.iloc[:,0:num_input].values
    train_features=torch.from_numpy(train_features).float()


    if output_style == 'energy_':
        train_labels = dt.iloc[:,num_input:num_input+1].values.reshape(-1, num_output)
    else:
        train_labels = dt.iloc[:,num_input+1:num_input+3].values.reshape(-1, num_output)
    
    StandardScaler = preprocessing.StandardScaler().fit(train_labels)
    RobustScaler = preprocessing.RobustScaler().fit(train_labels)

    if data_preprocessing_name == 'StandardScaler': 
        data_preprocessing = StandardScaler
        train_labels = data_preprocessing.transform(train_labels)

    if data_preprocessing_name == 'RobustScaler': 
        data_preprocessing = RobustScaler
        train_labels = data_preprocessing.transform(train_labels)

    if data_preprocessing_name == 'exp':
        train_labels = exp(train_labels)
    
    if data_preprocessing_name == 'e10':
        train_labels = e10(train_labels)
    
    train_labels = torch.from_numpy(train_labels).float()
    train_dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)

    if not os.path.exists(output_style  + '_StandardScaler.pkl'):
        pickle.dump(StandardScaler, open(output_style  + '_StandardScaler.pkl','wb'))
    if not os.path.exists(output_style  + '_RobustScaler.pkl'):
        pickle.dump(RobustScaler, open(output_style  + '_RobustScaler.pkl','wb'))    

    dt_test = pd.read_csv(testname)
    test_features=dt_test.iloc[:,0:num_input].values
    test_features=torch.from_numpy(test_features).float() 


    if output_style == 'energy_':
        test_labels = dt_test.iloc[:,num_input:num_input+1].values.reshape(-1, num_output)
    else:
        test_labels = dt_test.iloc[:,num_input+1:num_input+3].values.reshape(-1, num_output)

    if data_preprocessing_name == 'StandardScaler': 
        data_preprocessing = StandardScaler
        test_labels = data_preprocessing.transform(test_labels)

    if data_preprocessing_name == 'RobustScaler': 
        data_preprocessing = RobustScaler
        test_labels = data_preprocessing.transform(test_labels)

    if data_preprocessing_name == 'exp':
        test_labels = exp(test_labels)
    
    if data_preprocessing_name == 'e10':
        test_labels = e10(test_labels)

    test_labels = torch.from_numpy(test_labels).float()
    test_dataset = Data.TensorDataset(test_features, test_labels)
    test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=True)

    return train_iter, test_iter




def evaluate_accuracy(data_iter_def, model, loss, num_output, device=None):
    if device is None and isinstance(model, torch.nn.Module):
        device = list(model.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter_def:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y.view(-1, num_output))
            if isinstance(model, torch.nn.Module):
                model.eval()
                acc_sum += l.cpu().item()
                model.train() 
            n += y.shape[0]
    return acc_sum / n

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(7, 5)):
    plt.plot(x_vals,y_vals,label='Train Loss',linestyle='-',color='blue',lw=1.5)
    plt.plot(x2_vals,y2_vals,label='Test Loss',linestyle='--',color='red',lw=1.5)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc="upper right")
    
    

def use_svg_display():
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def train(model,train_iter_def, test_iter_def, loss_function, optimizer, epochs, loss, num_output):
    model = model.to(device)
    train_ls, test_ls = [], []
    for epoch in range(1, epochs + 1):
        start = time.time()    
        train_ls_epoch, n = 0.0, 0
        for X, y in train_iter_def:
            X = X.to(device)
            y = y.to(device)
            l = loss(model(X), y.view(-1, num_output))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls_epoch += l.cpu().item()
            n += y.shape[0]
        train_ls.append(train_ls_epoch / n)
        test_ls.append(evaluate_accuracy(test_iter_def, model,loss, num_output))
        #if (epoch % 10) == 0:
        print('epoch %d, train loss %.8f, test loss %.8f, time %.1f sec'% (epoch, train_ls[-1], test_ls[-1], time.time() - start))
    semilogy(range(1, epochs + 1), train_ls, 'epochs', 'loss',
             range(1, epochs + 1), test_ls, ['train', 'test'])
    return model, train_ls, test_ls

def predict(features, net, data_preprocessing_name=str, output_style = str):
    StandardScaler = pickle.load(open(output_style  +  '_StandardScaler.pkl', 'rb'))
    RobustScaler = pickle.load(open(output_style  +  '_RobustScaler.pkl', 'rb'))    

    if data_preprocessing_name == 'StandardScaler':
        result = StandardScaler.inverse_transform(net(features).cpu().detach().numpy().reshape(-1,2))
    elif data_preprocessing_name == 'RobustScaler':
        result = RobustScaler.inverse_transform(net(features).cpu().detach().numpy().reshape(-1,2))
    elif data_preprocessing_name == 'exp':
        result = inverse_exp(net(features).cpu().detach().numpy().reshape(-1,2))
    elif data_preprocessing_name == 'e10':
        result = inverse_e10(net(features).cpu().detach().numpy().reshape(-1,2))
    else:
        result = net(features).cpu().detach().numpy().reshape(-1,2)
    return result

