'''
Created on 3 dic 2020

@author: davidemiro
'''

import numpy as np
import jax.numpy as jnp
from PyLRAcluster.binary import *
from PyLRAcluster.poisson import *
from PyLRAcluster.gaussian import *
from PyLRAcluster.na import *
from PyLRAcluster.wsvd import gsvd
'''
from binary import *
from poisson import *
from gaussian import *
from na import *
'''
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

import random

def check_matrix_element(x):
    #in R una matrice è un tensore a due dimensioni
    if (len(x.shape) == 2):
        return False
    else:
        return True

def nrow_element(x):
    return x.shape[0]

def ncol_element(x):
    return x.shape[1]

def check(mat,type):
    if type == "binary":
        return check_binary(mat)
    elif type =="gaussian":
        return check_gaussian(mat)
    elif type =="poisson":
        return check_poisson(mat)
    else:
        return "Unknow type"




def LRAcluster(data,types,dimension=2,weights=None):


    eps = 0.0
    
    c = []
    i = 0
    for d in data:
        c.append(ncol_element(d))
    
    nSample = c[1]
    loglmin = 0
    loglmax = 0
    loglu = 0
    nData =len(data)
    nGeneArr = []
    for i in range(nData):
        data[i] = check(data[i],types[i])
        
    for d in data:
        nGeneArr.append(nrow_element(d))
    nGene = np.sum(np.array(nGeneArr))
    indexData = []
    k = 0
    for i in range(nData):


        indexData.append([j for j in range(k,k+nGeneArr[i])])
        k = k + nGeneArr[i]
 
    base = np.zeros((nGene,nSample))
    now = np.zeros((nGene,nSample))
    update = np.zeros((nGene,nSample))
    thr = np.zeros((nData,1))
    for i in range(nData):
        if types[i] == "binary":
            base[indexData[i],:] = base_binary(data[i])
            loglmin = loglmin +LLmin_binary(data[i],base[indexData[i],:])
            loglmax = loglmax + LLmax_binary(data[i])
        elif types[i] == "gaussian":
            base[indexData[i],:] = base_gaussian(data[i])
            loglmin = loglmin +LLmin_gaussian(data[i],base[indexData[i],:])
            loglmax = loglmax + LLmax_gaussian(data[i])
        elif types[i] == "poisson":
            base[indexData[i],:] = base_poisson(data[i])
            loglmin = loglmin +LLmin_poisson(data[i],base[indexData[i],:])
            loglmax = loglmax + LLmax_poisson(data[i])
    
    for i in range(nData):
        if types[i] == "binary":
            update[indexData[i],:] = update_binary(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))
        elif types[i] == "gaussian":
            update[indexData[i],:] = update_gaussian(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))
        elif types[i] == "poisson":
            update[indexData[i],:] = update_poisson(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))

    update = nuclear_approximation(update,dimension)
    nIter = 0
    thres = np.array([np.Inf,np.Inf,np.Inf])
    epsN = np.array([np.Inf,np.Inf])

    while True:
        
        for i in range(nData):
            if types[i] == "binary":
                thr[i] = stop_binary(data[i],base[indexData[i],:],now[indexData[i],:],update[indexData[i],:])
            elif types[i] == "gaussian":
                thr[i] = stop_gaussian(data[i],base[indexData[i],:],now[indexData[i],:],update[indexData[i],:])
            elif types[i] == "poisson":
                thr[i] = stop_poisson(data[i],base[indexData[i],:],now[indexData[i],:],update[indexData[i],:])
        print(nIter)
        nIter = nIter + 1
        
        thres[0] = thres[1]
        thres[1] = thres[2]
        thres[2] = sum(thr)
        epsN[0] = epsN[1]
        epsN[1] = eps
        if nIter > 5:

            if random.uniform(0,1) < thres[0]*thres[2]/(thres[1]*thres[1]+thres[0]*thres[2]):
                eps = epsN[0]+0.05*random.uniform(0,1) - 0.025
            else: 
                eps = epsN[1]+0.05*random.uniform(0,1) - 0.025
            if eps < -0.7:
                eps = 0
                epsN = np.array([0,0])
            if eps > 1.4:
                eps = 0
                epsN = np.array([0,0])
        if np.sum(thr) < nData*0.2:
            break
        now = np.copy(update)
        for i in range(nData):
            if types[i] == "binary":
                update[indexData[i],:] = update_binary(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))
            elif types[i] == "gaussian":
                update[indexData[i],:] = update_gaussian(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))
            elif types[i] == "poisson":
                update[indexData[i],:] = update_poisson(data[i],base[indexData[i],:],now[indexData[i],:],np.exp(eps))
   
        update = nuclear_approximation(update,dimension)

    for i in range(nData):
        if types[i] == "binary":
            loglu = loglu + LL_binary(data[i],base[indexData[i],:],update[indexData[i],:])
        elif types[i] == "gaussian":
            loglu = loglu + LL_gaussian(data[i],base[indexData[i],:],update[indexData[i],:])
        elif types[i] == "gaussian":
            loglu = loglu + LL_poisson(data[i],base[indexData[i],:],update[indexData[i],:])

    if weights ==None:
        u, s, v = randomized_svd(update,
                                n_components=dimension,
                                n_iter=5,
                                random_state=None)

    else:
        D1 = weights[0]
        D2 = weights[1]
        u, s, v = gsvd(update,np.diag(D1),np.diag(D2),dimension)

    coordinate = np.matmul(np.diag(s),v)
    ratio = (loglu -loglmin)/(loglmax - loglmin)
    return np.transpose(coordinate),ratio


#[m_illu,m_mirna,m_rna] = np.load("/Users/davidemiro/Desktop/speriamoPy.npy",allow_pickle=True)
#dma,a = LRAcluster([m_illu,m_mirna,m_rna],['gaussian','poisson','poisson'],2)





'''
clustering = KMeans(n_clusters=3, max_iter=500).fit(dma.reshape((430,2)))
clustering1 = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(dma.reshape((430,2)))
clustering2 = SpectralClustering(n_clusters=3, assign_labels='discretize').fit(dma.reshape((430,2)))


print("KMEANS")
print("silhuette: ", silhouette_score(dma.reshape((430,2)), clustering.labels_))
print("rand indexa: ", adjusted_rand_score(label, clustering.labels_))
print("AGGLOMERATIVE CLUSTERING")
print("silhuette: ", silhouette_score(dma.reshape((430,2)), clustering1.labels_))
print("rand indexa: ", adjusted_rand_score(label, clustering1.labels_))
print("SPECTRAL CLUSTERING")
print("silhuette: ", silhouette_score(dma.reshape((430,2)), clustering2.labels_))
print("rand indexa: ", adjusted_rand_score(label, clustering2.labels_))
'''

