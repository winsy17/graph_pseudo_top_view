#!/usr/bin/python

import numpy as np
from scipy import stats


'''This is a Leave one out classifier method.
input is a distance matrix X and a categorial function C. '''

def find_neighbours(number_neighbours, sample_index, dists):
    #output the indices of the nearest number_neighbours
    s = dists[sample_index] 
    sort_index = np.argsort(s) 
    return([x for x in sort_index[1:number_neighbours+1] if x!=sample_index])

def KNN(number_neighbours, sample_index, dists, Category_function):
    s = dists[sample_index] 
    sort_index = np.argsort(s) 
    nearest_neighbours = [x for x in sort_index[1:number_neighbours+1] if x!=sample_index]
    neighbour_categories = [Category_function[x] for x in nearest_neighbours]
    return(stats.mode(neighbour_categories)[0])
    
def Classification_rate(number_neighbours, dists, Category_function):
    n=dists.shape[0]
    correct=0.0
    for i in range(n):
        c=KNN(number_neighbours, i, dists, Category_function)
        if np.array_equal(c[0],Category_function[i]):
            correct+=1        
    return(correct/n)



if __name__ == '__main__':
    '''Example'''
    number_neighbours=5
    Y= np.array([(0,10,2,3),(10,0,2,2),(2,2,0,6),(3,2,6,0)])
    C=[100,200,200,200]
    category=KNN(3,2,Y,C)
    print(category)
    print(Classification_rate(3,Y,C))
    print(Classification_rate(1,Y,C))
    print(find_neighbours(2,0,Y))
    print(find_neighbours(2,1,Y))
    print(find_neighbours(2,2,Y))
    
    