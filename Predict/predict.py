import numpy.matlib
import numpy as np
import torch
import sys 
sys.path.append("..") 
import predict_function as func
from sklearn   import   preprocessing
import pickle
import random
import time
import pandas as pd
import multiprocessing
import math



if __name__ == '__main__':
    torch.set_num_threads(1)
    batch_size = 100000
    base_num = 5000
    
    start = time.time()

    operator = "ground_energy"    # ground_energy,rms_radius,quadrupole

    style = 'ANN'   # ANN,Brink
    if style == 'Brink':
        Kernel, norm = func.read_kernel_norm_from_Brink_matrix(base_num,operator)
        predicttime = 0
        print('Read %s bases GCM_Brink exact matrix element time required= %.2f seconds'%(str(base_num),time.time() - start))

    if style == 'ANN':
        if operator == 'ground_energy':
            Kernel, norm, predicttime = func.kernel_norm_ri_rj_exp(base_num,batch_size,operator)
            print('ANN predict %s bases matrix element time required = %.2f seconds'%(str(base_num),predicttime))
            func.write_KernelfromANN(Kernel,norm,base_num)  # write matrix element to file, optional
        if operator == 'rms_radius' or operator == 'quadrupole':
            Kernel, predicttime = func.kernel_norm_ri_rj_exp(base_num,batch_size,operator)
            norm = func.getANNnorm_numba(base_num)
        

    if operator == 'ground_energy':
        # begin GD
        random.seed(1)
        lr = 1
        eps = 0.00001
        max_iter_num = 100000
        c = []
        start = time.time()
        for i in range(base_num):
            c.append(random.normalvariate(0,1))
        c = np.array(c)
        energy_container = []
        GD_start = time.time()

        energy,energykernel,normkernel = func.energy(Kernel,norm,base_num,c)
        print('base_num = %5d,  energy =  %14.8f'%(base_num,energy))
        gradsumlast = {}
        gradsumlast[0]=0
        for step in range(0,max_iter_num):    
            gradsum = 0
            for i in range(0,base_num):
                grad = func.grad_energy_ground(Kernel, norm, energykernel,normkernel,base_num, c, i)
                c[i] -= lr*grad
                gradsum += abs(grad)
            
            if step > 10 and base_num !=50 and base_num != 100:
                lr = func.LearningRateScheduler(step, lr, gradsum, base_num, errorgrad)


            errorgrad = gradsum - gradsumlast[0]
            gradsumlast[0] = gradsum
            energy,energykernel,normkernel = func.energy(Kernel,norm,base_num,c)
            if step % 1 == 0:
                print('epochs = %6d, energy = %12.8f, gradsum = %12.8f, errorgrad = %12.8f, lr = %.1f, time = %6.1f'%(step,energy,gradsum,errorgrad,lr,time.time()-start))
            energy_container.append(energy)
            if step > 1:
                error = energy_container[step] - energy_container[step-1]
            if step > 50 and abs(error)<eps:
                max_iter_num = step
                break
            if step > 50 and errorgrad > 0:
                max_iter_num = step
                break

        func.write_coefficient_last(max_iter_num,base_num,c,energy,style)   # write the obtained coefficient to the file
        print('base_num = %5d,  energy =  %14.8f'%(base_num,energy))

        GD_end = time.time()
        # end GD


    

    result_predict=open('result_predict.csv','a+')

    result_predict.write('\n')
    if operator == 'ground_energy':
        item = '%6s, base_num %5d, ground energy %8.5f, Predict time %8.1fs, GD time %8.4f mins'%(style,base_num,energy,predicttime,(GD_end-GD_start)/60)
    elif operator == 'rms_radius':
        c = func.getcfromfile(base_num,style)
        energy,energykernel,normkernel = func.energy(Kernel,norm,base_num,c)
        item = '%6s, base_num %5d, rms_radius %8.5f, Predict time %8.1fs'%(style,base_num,math.sqrt(energy/8-3*1.35*1.35/16),predicttime)
    elif operator == 'quadrupole':
        c = func.getcfromfile(base_num,style)
        energy,energykernel,normkernel = func.energy(Kernel,norm,base_num,c)
        item = '%6s, base_num %5d, quadrupole %8.5f, Predict time %8.1fs'%(style,base_num,energy,predicttime)

    result_predict.write(item)

