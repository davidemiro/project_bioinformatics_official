'''
Created on 4 dic 2020

@author: davidemiro
'''

import numpy as np

from PyLRAcluster.na import *
#from na import *

epsilon = 2.0

def check_binary_row(arr):
    if np.sum(~ np.isnan(arr))==0:
        return False
    else:
        idx = ~ np.isnan(arr)
        if np.sum(arr[idx])==0 or np.sum(arr[idx])== np.sum(idx):
            return False
        else:
            return True

def check_binary(mat):
    index = np.apply_along_axis(check_binary_row, 1,mat)
    n = sum(~index)
    if n > 0:
        print("Invalid Lines")
    mat_c = mat[index,:]
    return mat_c
def base_binary_row(arr):
    
    idx = arr == 0
    n = np.sum(idx)
    m = np.sum(arr[idx])
    return np.log(m/(n-m))
    
def base_binary(mat):
    mat_b = np.zeros((mat.shape[0],mat.shape[1]))
    ar_b = np.apply_along_axis(base_binary_row, 1,mat)
    mat_b[0:mat_b.shape[0],:] = ar_b
    return mat_b

def update_binary(mat,mat_b,mat_now,eps):
    mat_p = mat_b+mat_now
    mat_u = np.zeros(mat.shape[0],mat.shape[1],mat) 
    idx1 = ~np.isnan(mat) & mat==1
    idx0 = ~np.isnan(mat) & mat==0
    index = np.isnan(mat)
    arr = np.exp(mat_p)
    mat_u[index] = mat_now[index]
    mat_u[idx1]<-mat_now[idx1]+eps*epsilon/(1.0+arr[idx1])
    mat_u[idx0]<-mat_now[idx0]-eps*epsilon*arr[idx0]/(1.0+arr[idx0])
    return mat_u

def stop_binary(mat,mat_b,mat_now,mat_u):
    index = ~np.isnan(mat)
    mn=mat_b+mat_now
    mu=mat_b+mat_u
    arn=np.exp(mn)
    aru=np.exp(mu)
    idx1 = ~np.isnan(mat) & mat==1
    idx0 = ~np.isnan(mat) & mat==0
    lgn=np.sum(np.log(arn[idx1]/(1+arn[idx1])))+np.sum(np.log(1/(1+arn[idx0])))
    lgu=np.sum(np.log(aru[idx1]/(1+aru[idx1])))+np.sum(np.log(1/(1+aru[idx0])))
    return lgu-lgn

def LL_binary(mat,mat_b,mat_u):
    index = ~np.isnan(mat)
    mu = mat_b +mat_u
    aru=np.exp(mu)
    idx1 = ~np.isnan(mat) & mat==1
    idx0 = ~np.isnan(mat) & mat==0
    lgu=np.sum(np.log(aru[idx1]/(1+aru[idx1])))+np.sum(np.log(1/(1+aru[idx0])))
    return lgu
def LLmax_binary(mat):
    return 0

def LLmin_binary(mat,mat_b):
    index = ~np.isnan(mat)
    aru = np.exp(mat_b)
    idx1 = ~np.isnan(mat) & mat==1
    idx0 = ~np.isnan(mat) & mat==0
    lgu=np.sum(np.log(aru[idx1]/(1+aru[idx1])))+np.sum(np.log(1/(1+aru[idx0])))
    return lgu

def binary_type_base(data,dimension=2 ,name="test"):
    data = check_binary(data, name)
    data_b = base_binary(data)
    data_now = np.zeros(data.shape[0],data.shape[1])
    data_u = update_binary(data,data_b,data_now)
    data_u = nuclear_approximation(data_u,dimension)
    while True:
        thr = stop_binary(data,data_b,data_now,data_u)
        if thr < 0.2:
            break
        data_now = data_u
        data_u = update_binary(data,data_b,data_now)
        data_u = nuclear_approximation(data_u,dimension)
    return data_now

    
    
    