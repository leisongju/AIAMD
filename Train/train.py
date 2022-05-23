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
#from IPython import display
import sys
import json
from sklearn   import   preprocessing
import pickle
import math
import copy
import function as func
#plt.style.use('science')


class Logger(object):
 
    def __init__(self, filename="Default.log"):
 
        self.terminal = sys.stdout
 
        self.log = open(filename, "a")
 
 
 
    def write(self, message):
 
        self.terminal.write(message)
 
        self.log.write(message)
 
 
 
    def flush(self):
 
        pass
path = os.path.abspath(os.path.dirname(__file__))
 
type = sys.getfilesystemencoding()
 
sys.stdout = Logger('record.txt')


torch.set_num_threads(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_output = 2
num_epochs = 100
data_num = 100

outname = ''
output_style = 'off-Diagonal'    # Diagonal  or  off-Diagonal
batch_size = 32
lr = 0.0001
weight_decay = 0.0001

loss = nn.L1Loss()

# StandardScaler  RobustScaler  exp  e10
data_preprocessing_name = 'RobustScaler'






if output_style == 'Diagonal':
    num_input = 6
    layer1 = 54
    layer2 = 54
    seed1=12
    func.setup_seed(seed1)
elif output_style == 'off-Diagonal':
    seed1=22
    func.setup_seed(seed1)
    num_input = 18
    layer1 = 252
    layer2 = 252


trainname = './data/train_' + output_style + '.csv'
testname = './data/test_' + output_style + '.csv'

train_iter, test_iter = func.DataLoader(trainname, testname, num_input, num_output, batch_size, output_style, data_preprocessing_name)



net = nn.Sequential(
    nn.Linear(num_input,layer1),
    nn.ReLU(),
    nn.Linear(layer1,layer2),
    nn.ReLU(),
    nn.Linear(layer2,num_output),
    )

print(net) 




print('lr = %12.8f, weight_decay = %12.8f' %(lr,weight_decay))
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)


    
if __name__== '__main__':
    modelname = output_style + '_' + data_preprocessing_name + '_epoch_' + str(num_epochs) + '_batchsize_'+str(batch_size) +'_'+str(loss) + '_lr_' + str(lr) +'_weightdecay_' + str(weight_decay) + outname
    
    print(modelname)

    start1 = time.time()
    net, train_ls, test_ls = func.train(net, train_iter, test_iter, loss, optimizer, num_epochs, loss, num_output)
    time_train = (time.time()-start1)/3600
    for i in range(1,100):
        name = output_style + str(i)
        if not os.path.exists(name):
            os.mkdir(name)
            model = './'+ name + '/' + modelname + '.pt'
            picture = './'+ name + '/' + modelname + '.pdf'
            losscsv = './'+ name + '/' + modelname + '.csv'
            torch.save(net, model)
            plt.savefig(picture)
            break



    print(model)
    f = {'epochs':list(range(1,num_epochs+1)),'train':train_ls,'test':test_ls}
    f = pd.DataFrame(f)
    f.to_csv(losscsv,index=False) 