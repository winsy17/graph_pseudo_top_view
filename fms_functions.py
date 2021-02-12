#!/usr/bin/env/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy 
import random
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise as pw
from sklearn.cluster import KMeans 
from sklearn.metrics.cluster import fowlkes_mallows_score as fms
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster import hierarchy 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [10, 10]
from Data_paths import path_fm_plot

clustering_method = 'complete'

'''Compute pairwise Euclidean distances given some vectors (presumably associated to the graphs)
In: Two arrays, each a set of feature vectors (shape m by n with m=number of graphs, n=feature vector dimension), and npoints = number of graphs
Out: Pairwise distance matrix'''
def pairwise_format(X, npoints):
    Xdist = pw.euclidean_distances(X)
    XX = Xdist[np.triu_indices(npoints,k=1)] #This k is diagonal offset
    return XX


'''Cluster data, compute and plot Fowlkes-Mallows score with mean and var 
In: Two pairwise distance matrices, npoints = number of points (graphs), plotname = title of plot, plotsave = name of plot to be saved 
Out: Plot of FM scores (Possibly also the dendrogram amd actual FM scores in some other format)
'''
def fms_compare(XX, YY, npoints, plot_title, plot_save):
    #Clustering
    ZXc = hierarchy.linkage(XX, method=clustering_method)
    ZYc = hierarchy.linkage(YY, method=clustering_method)

    #Cut dendrogram to obtain labelling for each k value
    #Warning: using hierarchy.cut_tree, but this function has a known bug!
    fms_dict = {}
    mean_dict = {}
    mean_dict[npoints]=0
    varbound_dict = {}
    varbound_dict[npoints]=0
    for i in range(1,npoints+1):
        ZXc_cut = [l for sublist in hierarchy.cut_tree(ZXc, i) for l in sublist]
        ZYc_cut = [l for sublist in hierarchy.cut_tree(ZYc, i) for l in sublist] 

        #Compute FM scores
        score = fms(ZXc_cut, ZYc_cut)
        fms_dict[i] = score

        #Compute moments for plotting and analysis
        c = contingency_matrix(ZXc_cut, ZYc_cut, sparse=True)
        tk = np.dot(c.data, c.data) - npoints
        pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - npoints
        qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - npoints
        pk2 = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 3) - 3*(np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2)) + 2*(np.sum(np.asarray(c.sum(axis=0)).ravel())) 
        qk2 = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 3) - 3*(np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2)) + 2*(np.sum(np.asarray(c.sum(axis=1)).ravel()))  
        if i < npoints:
            mean = (np.sqrt(pk*qk)) / (npoints*(npoints-1))
            mean_dict[i] = mean
            variance = (2/(npoints*(npoints-1))) + ((4*pk2*qk2)/(npoints*(npoints-1)*(npoints-2)*pk*qk))+ (((pk-2-((4*pk2)/pk))*(qk-2-((4*qk2)/qk)))/(npoints*(npoints-1)*(npoints-2)*(npoints-3))) - ((pk*qk)/((npoints**2)*((npoints-1)**2))) 
            varbound_dict[i] = 2* (variance**0.5)

    #Plot Bk and variance bounds
    lists = sorted(fms_dict.items())
    x, z = zip(*lists)
    upper = [mean_dict[i]+varbound_dict[i] for i in x]
    lower = [mean_dict[i]-varbound_dict[i] for i in x]
    means = [mean_dict[i] for i in x]

    #plt.plot(x,z)
    plt.scatter(x,z)
    plt.plot(x,upper)
    plt.plot(x, means)
    plt.plot(x,lower)
    plt.title(plot_title)
    plt.xlabel('# clusters')
    plt.ylabel('B_k')
    plt.savefig(path_fm_plot+ plot_save+'.jpg')
    plt.clf()

def fm_score(Xdist, Ydist, nclusters):
    ''' 
    Xdist and Ydist: two distance matrix of npoints by npoints
    nclusters: integer < npoints
    This function calculates the Fowlkes-Mallow scores between Xdist and Ydist with nclusters.'''
    #  Obtaining condensed distance matrices XX, YY.
    npoints = len(Xdist[0])
    XX = Xdist[np.triu_indices(npoints,k=1)]
    YY = Ydist[np.triu_indices(npoints,k=1)]
    #Clustering
    ZXc = hierarchy.linkage(XX, method=clustering_method)
    ZYc = hierarchy.linkage(YY, method=clustering_method)

    #Cut dendrogram to obtain labelling for nclusters
    ZXc_cut = [l for sublist in hierarchy.cut_tree(ZXc, nclusters) for l in sublist] 
    ZYc_cut = [l for sublist in hierarchy.cut_tree(ZYc, nclusters) for l in sublist] 
    score = fms(ZXc_cut, ZYc_cut)
    return(score)

def sil_list(Xdist, nclusters):
    ''' 
    Xdist: distance matrix of npoints by npoints
    nclusters: integer such that 2 <= ncusters< npoints
    Returns: list with silhouette scores of Xdist
    '''
    #  Obtaining condensed distance matrices XX
    npoints = len(Xdist[0])
    XX = Xdist[np.triu_indices(npoints,k=1)]
    #Clustering
    ZXc = hierarchy.linkage(XX, method=clustering_method)

    #Cut dendrogram to obtain labelling for each k value
    #Warning: using hierarchy.cut_tree, but this function has a known bug!
    sil = [-1]
    for i in range(2,nclusters+1):
        ZXc_cut = [l for sublist in hierarchy.cut_tree(ZXc, i) for l in sublist]
        cluster_labels = hierarchy.fcluster(ZXc, i, criterion='maxclust')
        sil += [round(silhouette_score(Xdist, cluster_labels, metric='precomputed'),3)]
    return(sil)

if __name__ == "__main__":
    '''Example'''
    #Sample for blob test
    X, sx = make_blobs(n_samples=100,
                  n_features=2,
                  centers=8,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True)

    Y, sy = make_blobs(n_samples=100,
                  n_features=2,
                  centers=8,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True)
    Xdist = pw.euclidean_distances(X)
    Ydist = pw.euclidean_distances(Y)
    print(fm_score(Xdist, Ydist, 4))
    print(sil_list(Xdist,20))
