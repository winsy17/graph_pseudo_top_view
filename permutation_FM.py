#!/usr/bin/python
import random
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np
from fms_functions import fm_score

'''This code computes a permutation test 
with the Fowlkes-Mallows as dependence measure between different metrics.
The input is two different distance matrices with the columns in the same order
The code computes the original Fowlkes-Mallows index
We then shuffle the labels (which graphs the columns correspond to) and recalculate the 
Fowlkes-Mallows index with the shuffled labels
If the metrics are independent then the rank of the original Fowlkes-Mallows index would be uniformly distributed
We can reject the independence hypothesis if the original Fowlkes-Mallows is 
consistently higher than other shuffled Fowlkes-Mallows'''

    
def compute_p_value_FM(orig_dists1, orig_dists2, number_of_permutations, nclusters):
    OriginalFM = fm_score(orig_dists1, orig_dists2, nclusters)
    N=orig_dists1.shape[0]
    rank=1.0
    Perm = list(range(N))   
    for k in range(number_of_permutations):
        random.shuffle(Perm)
        shuffled_dists2 = np.zeros((N,N))
        for i in range(N):
            for j in range(i):
                x=orig_dists2[Perm[i],Perm[j]]
                shuffled_dists2[i,j]=x
                shuffled_dists2[j,i]=x
        shuffledFM = fm_score(orig_dists1, shuffled_dists2, nclusters)
        if shuffledFM >= OriginalFM: # REJECT INDEPENDENCE
            rank+=1
    rank/=(number_of_permutations+1)
    return(rank)
    

    
if __name__ == '__main__':
    '''Example'''
    number_of_permutations = 199
    Y = np.array([(0,10,2,3),(10,0,2,2),(2,2,0,6),(3,2,6,0)])
    X = np.array([(0,9,6,4),(9,0,3,2),(6,3,0,5),(4,2,5,0)])
    p_value = compute_p_value_FM(X, Y, number_of_permutations,3)
    print(p_value)
    
    
    
    
    