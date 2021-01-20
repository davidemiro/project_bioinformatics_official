import numpy as np
from sklearn.utils.extmath import randomized_svd

import jax.numpy as jnp
import jax
from numpy.testing import assert_array_equal
import threading
from time import time
import scipy.linalg


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def do_dot(a, b, out):
    #np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    out[:] = np.dot(a, b)  # less efficient because the output is stored in a temporary array?


def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=a.dtype)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    threads = []
    for i in range(nblocks):
        for j in range(mblocks):
            th = threading.Thread(target=dot_func,
                                  args=(a_blocks[i, 0, :, :],
                                        b_blocks[0, j, :, :],
                                        out_blocks[i, j, :, :]))
            th.start()
            threads.append(th)

    for th in threads:
        th.join()

    return out

def matpower(X,n):
    tol = 1e-7
    nf = np.min(np.shape(X))


    u,s,v =jnp.linalg.svd(X)
    m = np.sum(s > tol)
    nf = np.min(np.array([nf,m]))
    x = np.matmul(np.matmul(u[:,:nf],np.diag(s[:nf]**n)),np.transpose(v[:,:nf]))
    return x



def wsvd(X,D1,D2,dimension):
    D1 = np.diag(D1)
    D2 = np.diag(D2)
    
    r =[D1,D2]
    p_d1 = matpower(D1, 0.5)
    X = pardot(p_d1,X,2,4)
    p_d2 = matpower(D2,0.5)
    X = pardot(X,p_d2,2,4)
    
    [u,d,v]  = jnp.linalg.svd(X)
    p_d1 = matpower(D1,-0.5)
    u = pardot(u,p_d1,2,4)
    p_d2 = matpower(D2,-0.5)
    v = pardot(p_d2,v,2,4)
    
    if np.all(u[:,1] < 0):
        u = u * -1
        v = v * -1
    
    
    return u,d,v

def gsvd(a, m, w,dimension):
	"""
	:param a: Matrix to GSVD
	:param m: 1st Constraint, (u.T * m * u) = I
	:param w: 2nd Constraint, (v.T * w * v) = I
	:return: (u ,s, v)
	"""

	(aHeight, aWidth) = a.shape
	(mHeight, mWidth) = m.shape
	(wHeight, wWidth) = w.shape

	#assert(aHeight == mHeight)
	#assert(aWidth == wWidth)

	mSqrt = np.sqrt(m)
	wSqrt = np.sqrt(w)


	mSqrtInv = jnp.linalg.inv(mSqrt).block_until_ready()
	wSqrtInv = jnp.linalg.inv(wSqrt).block_until_ready()

	_a = jnp.dot(jnp.dot(mSqrt, a).block_until_ready(), wSqrt).block_until_ready()

	(_u, _s, _v) = randomized_svd(_a,
                              n_components=dimension,
                              n_iter=5,
                              random_state=None)

	u = jnp.dot(mSqrtInv, _u).block_until_ready()
	v = jnp.dot(wSqrtInv, _v.T).block_until_ready().T.block_until_ready()
	s = _s

	return (np.copy(u), np.copy(s), np.copy(v))
    
    
    