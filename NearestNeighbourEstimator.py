#!/usr/bin/python

import numpy as np
from scipy import stats

'''Leave one out k nearest neighbours regression
This finds the k nearest neighbours and estimates a function 
from the average of the values from those of its neighbours

We then get a residual for each sample. 
This is the difference between the true function value and its estimate.

The final step is to calculate the mean square error (MSE) which is just the 
mean squared length of the residuals. 
MSE is always non-negative and close to 0 means a good estimation.

This can work with real valued or vector valued functions'''


def find_neighbours(number_neighbours, sample_index, dists):
    #output the indices of the nearest number_neighbours
    s = dists[sample_index] 
    sort_index = np.argsort(s) 
    return([x for x in sort_index[1:number_neighbours+1] if x != sample_index])

def Regression(number_neighbours, sample_index, dists, Function):
    s = dists[sample_index] 
    sort_index = np.argsort(s) 
    nearest_neighbours = [x for x in sort_index[1:number_neighbours+1] if x!=sample_index] 
    neighbour_values = [Function[x] for x in nearest_neighbours]
    return(np.mean(neighbour_values, axis = 0))
    
def Mean_Square_estimator(number_neighbours, dists, Function):
    n = dists.shape[0]
#    Estimate = [Regression]
    Residuals = [(Function[i]-Regression(number_neighbours, i, dists, Function)) for i in range(n)]
    Square_errors = [np.dot(Residuals[i],Residuals[i]) for i in range(n)]
    return(np.mean(Square_errors, axis = 0))     


if __name__ == '__main__':
    '''Example'''
    number_neighbours=5
    Y= np.array([(0,10,2,3),(10,0,2,2),(2,2,0,6),(3,2,6,0)])
    C1=[2,3,-1,4]
    C2=[[10,30],[0,0],[3,10],[10,20]]
    Regression(3,0,Y,C1)
    Regression(3,1,Y,C1)
    Regression(2,3,Y,C1)  
    mse=Mean_Square_estimator(number_neighbours, Y,C1) 
    print(mse)
    mse=Mean_Square_estimator(number_neighbours, Y,C2) 
    print(mse)
    print(find_neighbours(2,0,Y))
    print(find_neighbours(2,1,Y))
    print(find_neighbours(2,2,Y))
    