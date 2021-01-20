'''
Created on 5 dic 2020

@author: davidemiro
'''

import numpy as np
import jax.numpy as jnp
from sklearn.utils.extmath import randomized_svd

def nuclear_approximation(mat,dimension):
    

    u, s, v = randomized_svd(mat, 
                              n_components=dimension+1,
                              n_iter=5,
                              random_state=None)
    if dimension < s.shape[0]:
        lamb = s[dimension]
        indexh = s > lamb
        indexm = s < lamb
        s[indexh] = s[indexh] - lamb
        s[indexm] = 0
        mat_low = jnp.matmul(jnp.matmul(u[:,:dimension],np.diag(s)[:dimension,:dimension]).block_until_ready(),v[:dimension,:]).block_until_ready()
    else:
        mat_low = mat
    return np.copy(np.asarray(mat_low))


        
