#!/usr/bin/env/python3
import numpy as np
from sklearn.metrics import pairwise as pw
from math import sqrt as sq

'''This script computes sample distance covariance for a set of paired samples. See pg 8 of https://arxiv.org/pdf/1903.01051.pdf'''


def paired_sample_covar(a, b):
    #a,b pairwise distance matrices of equal size 
    npoints = len(a[0])
    #Compute means
    a_row_means = np.mean(a, axis=1)
    a_col_means = np.mean(a, axis=0)
    b_row_means = np.mean(b, axis=1)
    b_col_means = np.mean(b, axis=0)
    a_tot_mean = np.mean(a)  
    b_tot_mean = np.mean(b)

    #Compure dcov (and double centred matrix entries)
    products = []
    for k in range(npoints):
        for l in range(npoints):
            products.append((a.item((k,l)) - a_row_means.item(k) - a_col_means.item(l) + a_tot_mean)*(b.item((k,l)) - b_row_means.item(k) - b_col_means.item(l) + b_tot_mean)) 
    dcov = np.mean(np.array(products))
    return(sq(dcov))

def d_cor(a, b):
    #a,b pairwise distance matrices of equal size
    dvar_a =  paired_sample_covar(a,a)
    dvar_b = paired_sample_covar(b,b)
    if dvar_a*dvar_b == 0:
        return(0)
    dcor = (paired_sample_covar(a, b))/sq(dvar_a*dvar_b)
    return(dcor)

if __name__ == "__main__":
    npoints=3
    X = np.random.rand(npoints,2)
    Y = np.random.rand(npoints,2)
    #Pairwise distance matrices
    a = pw.euclidean_distances(X)
    b = pw.euclidean_distances(Y)
    dcov = paired_sample_covar(a,b)
    print(dcov)
