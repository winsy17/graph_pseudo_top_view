import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from fms_functions import * # to do hierarchical clustering and fawlkes mallows
from sdcovar_functions import * # to calculate distance correlations
from sklearn.metrics import pairwise as pw
from scipy.cluster import hierarchy 
from scipy.spatial.distance import squareform as sqf
from model_dictionary import *
from Data_paths import *
from PermutationDistCorr import compute_p_value 
from permutation_FM import compute_p_value_FM
from NearestNeighbourEstimator import Mean_Square_estimator
from LOOneighboursclassifier import Classification_rate

def matrix_slicing(model, metric):
    '''This function returns the distance matrix of all the parameters from model 
    model: list of strings, sublist of model_dict
    metric: string in metrics_dict or feat_dict'''
    file_name = metric + '.npy'
    D = np.load(path_dist + file_name)
    indices = []
    # slice distance matrix given model and parameter lists
    for string in model:
        temp = [i for i in range(model_dict.index(string)*graphs_per_param,(model_dict.index(string)+1)*graphs_per_param)]
        indices += temp
    # return sliced matrix
    return(D[np.ix_(sorted(indices),sorted(indices))])

def label_slicing(model, label):
    ''' model: list of strings, sublist of model_dict
    label: string 'parameter', 'Betti', 'simplex', 'family'
    '''
    if label == 'parameter' and dataset == 'Dataset':
        label_list = []
        for string in model:
            i = model_dict.index(string)
            label_list += [i]*graphs_per_param
    elif label == 'parameter' and dataset =='Dataset2':
        label_list = np.loadtxt(path_graphs + 'parameters')
        indices = []
        # slice parameter list given model
        for string in model:
            temp = [i for i in range(model_dict.index(string)*graphs_per_param,(model_dict.index(string)+1)*graphs_per_param)]
            indices += temp
        # return sliced parameter list summary as a list
        return(label_list[np.ix_(sorted(indices))])
    elif label == 'family' and dataset == 'Dataset':
        label_list = []
        for string in model:
            i = model_dict.index(string)
            if i <= 5:
                label_list += [0]*graphs_per_param
            elif i<= 8:
                label_list += [1]*graphs_per_param
            else:
                label_list += [2]*graphs_per_param
    elif label == 'family' and dataset == 'Dataset2':
        label_list = []
        for string in model:
            i = model_dict.index(string)
            label_list += [i]*graphs_per_param
    else:
        label_list = np.loadtxt(path_top + 'all_log_' + label)
        indices = []
        # slice topological summary given model and parameter lists
        for string in model:
            temp = [i for i in range(model_dict.index(string)*graphs_per_param,(model_dict.index(string)+1)*graphs_per_param)]
            indices += temp
        # return sliced topological summary as a list
        return(label_list[np.ix_(sorted(indices))])#.tolist())
    return(np.array(label_list))

def distance_correlation(model, metric_top, metric):
    '''model: list of strings, sublist of model_dict
    metric_top: string in feat_dict
    metric: string in metrics_dict'''
    D_top = matrix_slicing(model, metric_top)
    D_metric = matrix_slicing(model, metric)
    return(d_cor(D_top,D_metric))

def FM_plot(model, metric_top, metric):
    '''model: list of strings, sublist of model_dict
    metric_top: string in feat_dict
    metric: string in metrics_dict'''
    D_top = matrix_slicing(model, metric_top)
    D_metric = matrix_slicing(model, metric)
    num_graphs = len(D_top[0]) # number of graphs
    top_feat = metric_top.replace("top_","")
    file_name = top_feat + '_' + metric + '_' 
    for m in model:
        file_name += m +'_'
    fms_compare(sqf(D_top), sqf(D_metric), num_graphs, top_feat + ' vs ' + metric, file_name)
    plt.clf()

def FM_score(model, metric_top, metric, nclusters):
    '''model: list of strings, sublist of model_dict
    metric_top: string in feat_dict
    metric: string in metrics_dict'''
    D_top = matrix_slicing(model, metric_top)
    D_metric = matrix_slicing(model, metric)
    return(fm_score(D_top, D_metric,nclusters))

def cluster_dendogram(model,metric):
    '''model: list of strings, sublist of model_dict
    metric is a string: 'PD', 'TEuc', 'TEMD', 'random', 'top_log_Betti' or 'top_log_simplex'
    '''
    #obtain sliced distance matrix 
    D = matrix_slicing(model, metric)
    ZXc = hierarchy.linkage(sqf(D), method='complete')
    title = metric
    for m in model:
        title += '_' + m
    # Calculate dendogram
    plt.figure(figsize=(25, 10))
    plt.title(title)
    plt.xlabel('sample index')
    plt.ylabel(metric + '_distance')
    hierarchy.dendrogram(ZXc,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig(path_dendo +title+'.jpg')
    plt.clf()

def list_sil(model, metric):
    ''' model: list of strings, sublist of model_dict
    metric_top: string in feat_dict
    metric: string in metrics_dict'''
    D = matrix_slicing(model, metric)
    return(sil_list(D,len(D[0])))

def p_val_corr(model,metric_1,metric_2, npermutations):
    ''' model: list of strings, sublist of model_dict
    metric_1: string in feat_dict or metrics_dict
    metric_2: string in feat_dict or metrics_dict
    npermutations: number of permutations'''
    D_1 = matrix_slicing(model, metric_1)
    D_2 = matrix_slicing(model, metric_2)
    return(compute_p_value(D_1,D_2,npermutations))

def p_val_FM(model,metric_1,metric_2, npermutations,nclusters):
    ''' model: list of strings, sublist of model_dict
    metric_1: string in feat_dict or metrics_dict
    metric_2: string in feat_dict or metrics_dict
    npermutations: number of permutations'''
    D_1 = matrix_slicing(model, metric_1)
    D_2 = matrix_slicing(model, metric_2)
    return(compute_p_value_FM(D_1,D_2,npermutations,nclusters))


def MSE(model, metric, nneighbours, target_label):
    '''
    model: list of strings, sublist of model_dict
    metric: string in metrics_dict or feat_dict
    target_label = : string 'family', 'Betti', 'simplex'
    '''
    D = matrix_slicing(model, metric)
    targets = label_slicing(model, target_label)
    return(Mean_Square_estimator(nneighbours, D, targets))

def rate(model, metric, nneighbours, target_label):
    '''
    model: list of strings, sublist of model_dict
    metric: string in metrics_dict or feat_dict
    target_label = : string 'family', 'Betti', 'simplex'
    '''
    D = matrix_slicing(model, metric)
    targets = label_slicing(model, target_label)
    return(Classification_rate(nneighbours, D, targets))

if __name__ == "__main__":
    models = model_dict
    top_feat = feat_dict 
    metrics = metrics_dict
    print(list_sil(models,'PD'))
    print(p_val_corr(model_dict,metrics[0],top_feat[0],2000))
    print(p_val_FM(model_dict,metrics[0],top_feat[0],200,6))
    
    

    