#!/usr/bin/python

import random
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np

'''This code computes a permutation test 
with the distance correlation for dependence between different metrics.
The input is two different distance matrices with the columns in the same order
The code computes the original distance covariance 
We then shuffle the labels (which graphs the columns correspond to) and recalculate the 
distance correlation with the shuffled labels
If the metrics are independent 
then the rank of the original correlation is uniformly distributed
We can reject the independence hypothesis if the original distance correlation is 
consistently higher than other shuffled distance correlations'''


def distcorr(X, Y):
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    A = X - X.mean(axis = 0)[None, :] - X.mean(axis = 1)[:, None] + X.mean()
    B = Y - Y.mean(axis = 0)[None, :] - Y.mean(axis = 1)[:, None] + Y.mean()
    dcov2_xy = (A * B).sum()
    dcov2_xx = (A * A).sum()
    dcov2_yy = (B * B).sum()
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return(dcor)

def distCov2(X, Y):
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    A = X - X.mean(axis = 0)[None, :] - X.mean(axis = 1)[:, None] + X.mean()
    B = Y - Y.mean(axis = 0)[None, :] - Y.mean(axis = 1)[:, None] + Y.mean()
    return((A * B).sum())
    
def compute_p_value(orig_dists1, orig_dists2, number_of_permutations):
    OriginalDCov2 = distCov2(orig_dists1, orig_dists2)
    N = orig_dists1.shape[0]
    rank = 1.0
    Perm = list(range(N))       
    for k in range(number_of_permutations):
        random.shuffle(Perm)
        shuffled_dists2 = np.zeros((N,N))
        for i in range(N):
            for j in range(i):
                x = orig_dists2[Perm[i],Perm[j]]
                shuffled_dists2[i,j] = x
                shuffled_dists2[j,i] = x
        shuffledDCov2 = distCov2(orig_dists1, shuffled_dists2)
        if shuffledDCov2 >= OriginalDCov2:
            rank+=1    
    rank/=(number_of_permutations+1)
    return(rank)
    

    
if __name__ == '__main__':
    '''Example'''
    number_of_permutations = 199
    Y = np.array([(0,10,2,3),(10,0,2,2),(2,2,0,6),(3,2,6,0)])
    X = np.array([(0,9,6,4),(9,0,3,2),(6,3,0,5),(4,2,5,0)])
    p_value = compute_p_value(X, Y, number_of_permutations)
    print(p_value)
    
    
    
    
    