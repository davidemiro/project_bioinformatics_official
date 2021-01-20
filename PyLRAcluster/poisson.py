'''
Created on 4 dic 2020

@author: davidemiro
'''
import numpy as np
from PyLRAcluster.na import *
#from na import *
epsilon_poisson=0.5

def check_poisson_row(arr):
    if np.sum(~ np.isnan(arr))==0:
        return False
    else:
        idx = ~ np.isnan(arr)
        if np.sum(arr[idx]<0)>0: 
            return False
        else:
            return True

def check_poisson(mat):
    index = np.apply_along_axis(check_poisson_row, 1,mat)
    n = np.sum(~index)
    if n > 0:
         print("Errore")
         
    mat_c = mat[index,:] + 1
    return mat_c
def base_poisson_row(arr):
    idx = ~np.isnan(arr)
    m = np.sum(np.log(arr[idx]))
    n = np.sum(idx)
    return m/n
def base_poisson(mat):
    mat_b = np.zeros((mat.shape[0],mat.shape[1]))
    ar_b = np.apply_along_axis(base_poisson_row, 1,mat)
    for i in range(mat.shape[1]):
        mat_b[:,i] = ar_b
    return mat_b

def update_poisson(mat,mat_b,mat_now,eps):
    mat_p =mat_b + mat_now
    mat_u =np.zeros((mat.shape[0],mat.shape[1]))
    index = ~np.isnan(mat)
    mat_u[index] = mat_now[index]+eps*epsilon_poisson*(np.log(mat[index])-mat_p[index])
    index=~np.isnan(mat)
    mat_u[index]<-mat_now[index]
    return mat_u

def stop_poisson(mat,mat_b,mat_now,mat_u):
    index = ~np.isnan(mat)
    mn = mat_b + mat_now
    mu = mat_b + mat_u
    lgn = np.sum(mat[index]*mn[index]-np.exp(mn[index]))
    lgu = np.sum(mat[index]*mu[index]-np.exp(mu[index]))
    return lgu - lgn

def LL_poisson(mat,mat_b,mat_u):
    index = ~np.isnan(mat)
    mu = mat_b + mat_u
    lgu = np.sum(mat[index]*mu[index]-np.exp(mu[index]))
    return lgu

def LLmax_poisson(mat):
    index = ~np.isnan(mat)
    lgu = np.sum(mat[index]*np.log(mat[index])-mat[index])
    return lgu
def LLmin_poisson(mat,mat_b):
    index = ~np.isnan(mat)
    lgu = np.sum(mat[index]*mat_b[index]-np.exp(mat_b[index]))
    return lgu
def poisson_type_base(data,dimension=2,name="test"):
    data = check_poisson(data,name)
    data_b = base_poisson(data)
    data_now =np.zeros((data.shape[0],data.shape[1]))
    data_u = update_poisson(data, data_b, data_now, 1)
    data_u = nuclear_approximation(data_u,dimension)
    while True:
        thr = stop_poisson(data, data_b, data_now,data_u)
        if thr < 0.2:
            break
        data_now=data_u
        data_u = update_poisson(data, data_b, data_now, 1)
        data_u = nuclear_approximation(data_u,dimension)
    return data_now
    
    
            