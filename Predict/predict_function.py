import numpy.matlib
import numpy as np
import torch
from sklearn   import   preprocessing
import pickle
import math
import random
import time
import pandas as pd
import sys 
from numpy import genfromtxt
from numba import jit



@jit(nopython=True)
def read_kernel_norm_from_brink_matrix_numba(base_num,matrix):
    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))
    k=-1
    for i in range(base_num):
        for j in range(base_num):
            if i<=j:
                k+=1
                Kernel[i,j] = matrix[k,2]
                norm[i,j] = matrix[k,3]
    for i in range(base_num):
        for j in range(base_num):
            if i>j:
                Kernel[i,j] = Kernel[j,i]
                norm[i,j] = norm[j,i]
    return Kernel,norm

def read_kernel_norm_from_Brink_matrix(base_num,operator):
    matrix = get_gcmbrinkmatrix_operator(base_num,operator)
    Kernel, norm = read_kernel_norm_from_brink_matrix_numba(base_num,matrix)
    return Kernel, norm

def get_gcmbrinkmatrix_operator(base_num,operator):
    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))
    namematrix = './'+str(base_num) + '/Brink_matrix_'+operator+'.csv'
    #matrix = matrix[:,2:4]
    with open(namematrix,encoding = 'utf-8') as f:
        matrix = np.loadtxt(f,delimiter=',',skiprows=1)
    return matrix


def write_KernelfromANN(Kernel,norm,base_num):
    name = './'+str(base_num) + '/base_num_' + str(base_num) +'_ANN_predict_kernel_norm.csv'
    f=open(name,'w')
    f.write('bra,ket,ahb,norm')
    f.write('\n')
    for i in range(base_num):
        for j in range(base_num):
            if (i<=j):
                f.write(str(i+1))
                f.write(',')
                f.write(str(j+1))
                f.write(',')
                f.write(str(Kernel[i,j]))
                f.write(',')
                f.write(str(norm[i,j]))
                f.write('\n')

@jit(nopython=True) #Calculate the energy (or other physical quantity) according to the corresponding matrix element
def energy(Kernel,norm,base_num,c):
    energykernel = 0.0
    normkernel = 0.0
    for i in range(base_num):
        for j in range(base_num):
            if i == j:
                energykernel += c[i]*c[j]*Kernel[i,j]
                normkernel += c[i]*c[j]*norm[i,j]
            elif i<j:
                energykernel += 2*c[i]*c[j]*Kernel[i,j]
                normkernel += 2*c[i]*c[j]*norm[i,j]
    energy = energykernel/normkernel
    return energy,energykernel,normkernel


@jit(nopython=True) # Find a gradient for a given coefficient
def grad_energy_ground(Kernel,norm,energykernel,normkernel,base_num,c,k):
    delatc = 0.00001
    Dkernel, Dnorm = 0.0, 0.0
    for i in range(base_num):
        Dkernel += 2 * c[i]*Kernel[i,k]
        Dnorm += 2 * c[i]*norm[i,k]   
    grad = (Dkernel*normkernel)/(normkernel*normkernel) - ((Dnorm*energykernel)/(normkernel*normkernel))
    return grad



def LearningRateScheduler(epochs, lr, gradsum, base_num, errorgrad):  #Adjust learning rate
    lr1 = abs(math.log10(abs(errorgrad)))
    lr1 = 1
    lr2_1 = gradsum * 10
    lr2_2 = 10

    if base_num == 200 or base_num == 500 or base_num == 1000 or base_num == 2000:
        lr1 = 5
    
    if epochs > 10:
        if base_num == 5000:
            lr2_1 = gradsum * 10
            lr2_2 = 10
            lr1 = 5
        if base_num == 10000 or base_num == 20000 or base_num == 30000 or base_num == 40000 or base_num == 50000 or base_num == 100000:
            lr2_1 = gradsum * 10
            lr2_2 = 15
            lr1 = 10

    if epochs > 20:
        if base_num == 2000:
            lr2_1 = gradsum * 10
            lr2_2 = 20
            lr1 = 10
        if base_num == 5000:
            lr2_1 = gradsum * 100
            lr2_2 = 50
            lr1 = 25
        if base_num == 10000 or base_num == 20000 or base_num == 30000 or base_num == 40000 or base_num == 50000 or base_num == 60000 or base_num == 70000 or base_num == 80000 or base_num == 90000 or base_num == 100000:
            lr2_1 = gradsum * 100
            lr2_2 = 50
            lr1 = 25
    
    if epochs > 30:
        if base_num == 5000:
            lr2_1 = gradsum * 100
            lr2_2 = 50
            lr1 = 15
        if base_num == 10000 or base_num == 20000 or base_num == 30000 or base_num == 40000 or base_num == 50000 or base_num == 60000 or base_num == 70000 or base_num == 80000 or base_num == 90000 or base_num == 100000:
            lr2_1 = gradsum * 100
            lr2_2 = 200
            lr1 = 150
    lr2 = min(lr2_1,lr2_2)
    lr = max(lr1,lr2)
    return lr

def write_coefficient(step,base_num,c,energy,style):  #Write the obtained wave function coefficients into the file
    name = './'+str(base_num) + '/base_num_' + str(base_num) +'_'+style+'_quan.csv'
    fquan=open(name,'a')
    if step % 100 == 0:
        fquan.write('i,c')
        fquan.write('\n')
        for j in range(base_num):
            fa=str(j)+','+str(c[j])
            fquan.write(fa)
            fquan.write('\n')
        fquan.write('step = ')
        fquan.write(str(step))
        fquan.write('\n')
        fquan.write('energy = ')
        fquan.write(str(energy))
        fquan.write('\n')

def getANNnorm_numba(base_num,):
    matrix = get_ANNmatrix(base_num)
    Kernel, norm = getkernelnormformANNmatrix_numba(base_num,matrix)
    return norm

@jit(nopython=True)
def getkernelnormformANNmatrix_numba(base_num,matrix):
    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))
    k=-1
    for i in range(base_num):
        for j in range(base_num):
            if i<=j:
                k+=1
                Kernel[i,j] = matrix[k,2]
                norm[i,j] = matrix[k,3]
    for i in range(base_num):
        for j in range(base_num):
            if i>j:
                Kernel[i,j] = Kernel[j,i]
                norm[i,j] = norm[j,i]
    return Kernel,norm

def get_ANNmatrix(base_num):
    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))
    namematrix = './'+str(base_num) + '/base_num_' + str(base_num) +'_ANN_predict_kernel_norm.csv'
    #matrix = matrix[:,2:4]
    with open(namematrix,encoding = 'utf-8') as f:
        matrix = np.loadtxt(f,delimiter=',',skiprows=1)
    return matrix


def getcfromfile(base_num,style):
    namecsv = './'+str(base_num) + '/base_num_' + str(base_num) +'_'+ style + '_coefficient.csv'
    coefficient = pd.read_csv(namecsv)
    c = []
    for i in range(base_num):
        c.append(coefficient.iloc[i,1])
    c = np.array(c)
    return c

def bareqket_fromcsv_ri_rj_exp(base_num):
    name = './'+str(base_num) + '/bases.csv'
    f = pd.read_csv(name)
    base = []
    for i in range(base_num):
        base.append(f.iloc[i,0:9])
    base = np.array(base)
    base_ri_rj = []
    for i in range(len(f)):
        r1_r2 = (((f.iloc[i,0]-f.iloc[i,3])**2 + (f.iloc[i,1]-f.iloc[i,4])**2 + (f.iloc[i,2]-f.iloc[i,5])**2))
        r1_r3 = (((f.iloc[i,0]-f.iloc[i,6])**2 + (f.iloc[i,1]-f.iloc[i,7])**2 + (f.iloc[i,2]-f.iloc[i,8])**2))
        r2_r3 = (((f.iloc[i,3]-f.iloc[i,6])**2 + (f.iloc[i,4]-f.iloc[i,7])**2 + (f.iloc[i,5]-f.iloc[i,8])**2))
        
        r1_r2_exp = math.exp(-r1_r2)
        r1_r3_exp = math.exp(-r1_r3)
        r2_r3_exp = math.exp(-r2_r3)
        tmp = [r1_r2, r1_r3, r2_r3, r1_r2_exp, r1_r3_exp, r2_r3_exp]
        base_ri_rj.append(tmp)
    base_ri_rj = np.array(base_ri_rj)
    return base, base_ri_rj

@jit(nopython=True)
def ri_rj_generater_exp(basei,basej):
    r1b_r1k = (((basei[0]-basej[0])**2 + (basei[1]-basej[1])**2 + (basei[2]-basej[2])**2))
    r1b_r2k = (((basei[0]-basej[3])**2 + (basei[1]-basej[4])**2 + (basei[2]-basej[5])**2))
    r1b_r3k = (((basei[0]-basej[6])**2 + (basei[1]-basej[7])**2 + (basei[2]-basej[8])**2))
    r2b_r1k = (((basei[3]-basej[0])**2 + (basei[4]-basej[1])**2 + (basei[5]-basej[2])**2))
    r2b_r2k = (((basei[3]-basej[3])**2 + (basei[4]-basej[4])**2 + (basei[5]-basej[5])**2))
    r2b_r3k = (((basei[3]-basej[6])**2 + (basei[4]-basej[7])**2 + (basei[5]-basej[8])**2))
    r3b_r1k = (((basei[6]-basej[0])**2 + (basei[7]-basej[1])**2 + (basei[8]-basej[2])**2))
    r3b_r2k = (((basei[6]-basej[3])**2 + (basei[7]-basej[4])**2 + (basei[8]-basej[5])**2))
    r3b_r3k = (((basei[6]-basej[6])**2 + (basei[7]-basej[7])**2 + (basei[8]-basej[8])**2))


    r1b_r1k_exp = math.exp(-(r1b_r1k))
    r1b_r2k_exp = math.exp(-(r1b_r2k))
    r1b_r3k_exp = math.exp(-(r1b_r3k))
    r2b_r1k_exp = math.exp(-(r2b_r1k))
    r2b_r2k_exp = math.exp(-(r2b_r2k))
    r2b_r3k_exp = math.exp(-(r2b_r3k))
    r3b_r1k_exp = math.exp(-(r3b_r1k))
    r3b_r2k_exp = math.exp(-(r3b_r2k))
    r3b_r3k_exp = math.exp(-(r3b_r3k))

    base_off_Diagonal = np.array([r1b_r1k, r1b_r2k, r1b_r3k, r2b_r1k, r2b_r2k, r2b_r3k, r3b_r1k, r3b_r2k, r3b_r3k, r1b_r1k_exp, r1b_r2k_exp, r1b_r3k_exp, r2b_r1k_exp, r2b_r2k_exp, r2b_r3k_exp, r3b_r1k_exp, r3b_r2k_exp, r3b_r3k_exp])
    return base_off_Diagonal


def get_off_Diagnoal_base(base_num,batch_size,off_Diagonal_num):
    base, base_ri_rj = bareqket_fromcsv_ri_rj_exp(base_num)
    base_off_Diagonal = featrue_enhancement(base,base_num,off_Diagonal_num)
    base_off_Diagonal=torch.from_numpy(base_off_Diagonal).float()
    return base_off_Diagonal

@jit(nopython=True)
def featrue_enhancement(base,base_num,off_Diagonal_num):
    base_off_Diagonal=np.empty(shape=(off_Diagonal_num,18))
    k=0
    for i in range(0,base_num):
        for j in range(0,base_num):
            if i<j:
                base_off_Diagonal[k]=ri_rj_generater_exp(base[i],base[j])
                k+=1
    return base_off_Diagonal

def ann_predict_off_Diagonal(data_process_off_Diagonal,net_off_Diagonal,base_off_Diagonal,output):
    output_off_Diagonal = data_process_off_Diagonal.inverse_transform(net_off_Diagonal(base_off_Diagonal).cpu().detach().numpy().reshape(-1,output))
    return output_off_Diagonal

def ann_predict_off_Diagonal_operator(data_process_off_Diagonal,net_off_Diagonal,base_off_Diagonal):
    output_off_Diagonal = data_process_off_Diagonal.inverse_transform(net_off_Diagonal(base_off_Diagonal).cpu().detach().numpy().reshape(-1,1))
    return output_off_Diagonal

def kernel_norm_ri_rj_exp(base_num,batch_size,operator=str):  #Prediction of wave function matrix elements using neural networks
    if operator == 'ground_energy':
        output = 2
    elif operator == 'rms_radius':
        output = 1
    elif operator == 'quadrupole':
        output = 1

    start = time.time()
    off_Diagonal_num = (base_num * (base_num - 1))/2
    batch_num = int(off_Diagonal_num / batch_size)+1
    off_Diagonal_num = batch_size * batch_num

    base, base_Diagonal = bareqket_fromcsv_ri_rj_exp(base_num)
    base_Diagonal=torch.from_numpy(base_Diagonal).float()

    base_off_Diagonal=get_off_Diagnoal_base(base_num,batch_size,off_Diagonal_num)
    print('Time consumed for read and enhance',base_num,'bases',time.time()-start)

    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))

    Diagonal_name = './RobustScaler_and_net/'+ operator +'/Diagonal_RobustScaler_epoch_100_batchsize_32_L1Loss()_lr_0.0001_weightdecay_0.0001.pt'
    off_Diagonal_name = './RobustScaler_and_net/'+ operator +'/off_Diagonal_RobustScaler_epoch_100_batchsize_32_L1Loss()_lr_0.0001_weightdecay_0.0001.pt'
    net_Diagonal = torch.load(Diagonal_name,map_location=torch.device('cpu'))
    net_off_Diagonal = torch.load(off_Diagonal_name,map_location=torch.device('cpu'))
    data_process_Diagonal = pickle.load(open('./RobustScaler_and_net/'+ operator +'/Diagonal_RobustScaler.pkl', 'rb'))
    data_process_off_Diagonal = pickle.load(open('./RobustScaler_and_net/'+ operator +'/off-Diagonal_RobustScaler.pkl', 'rb'))
    

    output_off_Diagonal = np.empty(shape=(batch_size,batch_num,output))

    for i in range(batch_num):
        off_Diagonal_batch = base_off_Diagonal[i*batch_size:(i+1)*batch_size,:]
        output_off_Diagonal[:,i,:] = ann_predict_off_Diagonal(data_process_off_Diagonal, net_off_Diagonal, off_Diagonal_batch,output)
    

    Kernel = np.empty(shape=(base_num,base_num))
    norm = np.empty(shape=(base_num,base_num))
    for i in range(0,base_num):
        output_Diagonal = data_process_Diagonal.inverse_transform(net_Diagonal(base_Diagonal[i]).cpu().detach().numpy().reshape(-1,output))
        Kernel[i,i] = output_Diagonal[0,0]
        if output == 2:
            norm[i,i] = output_Diagonal[0,1]

    print('Time consumed for ANN predict',base_num,'bases (include the time for feature enhancement) matrix element',time.time()-start)
    
    predicttime = time.time() - start


    output_off_Diagonal_k_n = np.empty(shape=(off_Diagonal_num,output))
    for i in range(batch_num):
        output_off_Diagonal_k_n[i*batch_size:(i+1)*batch_size,:] = output_off_Diagonal[:,i,:]
    k = 0
    for i in range(0,base_num):
        for j in range(0,base_num):
            if i != j and i<j:
                Kernel[i,j] = output_off_Diagonal_k_n[k,0]
                if output == 2:
                    norm[i,j] = output_off_Diagonal_k_n[k,1]
                k += 1
            if i != j and i>j:
                Kernel[i,j] = Kernel[j,i]
                if output == 2:
                    norm[i,j] = norm[j,i]
    if output == 2:
        return Kernel, norm, predicttime
    if output == 1:
        return Kernel, predicttime



def write_coefficient_last(max_iter_num,base_num,c,energy,style):
    namecsv = './'+str(base_num) + '/base_num_' + str(base_num) +'_'+style +'_coefficient.csv'
    fc=open(namecsv,'w')
    fc.write('i,c')
    fc.write('\n')
    for j in range(base_num):
        fa2=str(j)+','+str(c[j])
        fc.write(fa2)
        fc.write('\n')