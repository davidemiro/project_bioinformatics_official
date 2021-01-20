'''
Created on 4 dic 2020

@author: davidemiro
'''

import numpy as np
from PyLRAcluster.na import *
#from na import *

epsilon_gaussian = 0.5

def check_gaussian_row(arr):
    if np.sum(~ np.isnan(arr))==0:
        return False
    else:
        return True

def check_gaussian(mat):
    index = np.zeros(mat.shape[0]) ==0
    for i in range(mat.shape[0]):
        if np.sum(np.isnan(mat[i,]))==mat.shape[1]:
            index[i] = False
    mat_c =mat[index,:]
    return mat_c
    
def base_gaussian_row(arr):
    return np.mean(arr)

def base_gaussian(mat):
    mat_b =np.zeros((mat.shape[0],mat.shape[1]))
    ar_b = np.apply_along_axis(base_gaussian_row, 1,mat)

    for i in range(mat.shape[1]):
        mat_b[:,i] = ar_b
    return mat_b
def update_gaussian(mat,mat_b,mat_now,eps):
    mat_p = mat_b +mat_now
    mat_u = np.zeros((mat.shape[0],mat.shape[1]))
    index = ~np.isnan(mat)
    mat_u = mat_now +eps*epsilon_gaussian*(mat - mat_p)
    index = np.isnan(mat)
    mat_u[index]=mat_now[index]
    return mat_u

def stop_gaussian(mat,mat_b,mat_now,mat_u):
    index = ~np.isnan(mat)
    mn=mat_b+mat_now
    mu=mat_b+mat_u
    ren=mat[index]-mn[index]
    reu=mat[index]-mu[index]
    lgn= -0.5*np.sum(ren*ren)
    lgu= -0.5*np.sum(reu*reu)
    return lgu-lgn

def LL_gaussian(mat,mat_b,mat_u):
    index = ~np.isnan(mat)
    mu=mat_b+mat_u
    reu=mat[index]-mu[index]
    lgu= -0.5*np.sum(reu*reu)
    return lgu

def LLmax_gaussian(mat):
    return 0.0

def LLmin_gaussian(mat,mat_b):
    index = ~np.isnan(mat)
    reu = mat[index] - mat_b[index]
    lgu = -0.5*sum(reu*reu)
    return lgu

def gaussian_base(data,dimension=2,name="test"):
    data_b = base_gaussian(data)
    data_now = np.zeros((data.shape[0],data.shape[1]))
    data_u = update_gaussian(data,data_b,data_now)
    data_u = nuclear_approximation(data_u,dimension)
    while True:
        thr = stop_gaussian(data,data_b,data_now,data_u)
        if thr < 0.2:
            break
        data_now=data_u
        data_u=update_gaussian(data,data_b,data_now)
        data_u=nuclear_approximation(data_u,dimension)
    return data_now
        
      
    