import numpy as np
import random
from Data_paths import *
from model_dictionary import dataset

def random_distance(s=120, _seed=42):
    '''generate a random distance matrix of size s:
    positive definite matrix with zeros in the diagonal
    '''
    random.seed(_seed)
    r_dist= np.zeros((s,s))
    for i in range(s):
        for j in range(i):
            r_dist[i][j] = random.random()
            r_dist[j][i] = r_dist[i][j]
    file_name = 'random.npy'
    np.save(path_dist + file_name, np.matmul(r_dist,r_dist.transpose()))
    return np.matmul(r_dist,r_dist.transpose())

if __name__ == "__main__":
    if dataset == 'Dataset':
        s = 120
    if dataset == 'Dataset2':
        s= 300
    random_distance(s)

