import networkx as nx
import pyflagser as pf
import random
import numpy as np
import scipy as sp
import time
import pickle
from Data_paths import path_graphs, path_top

def top_summary(graphs,dim = 7,approx = None):
    '''
    graphs: list of networkx directed graphs
    dim: maximum dimension to calculate Betti numbers and simplex counts
    approx: integer that determines the approximation used in pyflagser
    This function returns a list containing the Betti numbers up to dimension dim
    and another list containing the simplex counts up to dimension dim
    '''
    list_Betti = []
    list_simplex = []
    i = 0
    print('Generating Topological Summary')
    for g in graphs:
        resbetti = np.zeros((dim,))
        rescell = np.zeros((dim,))
        G_matrix = nx.to_scipy_sparse_matrix(g) #returns scipy sparse matrix
        out = pf.flagser_weighted(G_matrix + sp.sparse.identity(nx.number_of_nodes(g)),
                                min_dimension=0,
                                max_dimension=dim,
                                directed=True, approximation= approx)
        betti = out['betti']
        cell_count = out['cell_count']
        for i in range(dim):
            if i < len(betti):
                resbetti[i] = betti[i]
            else:
                resbetti[i] = 0

        for i in range(dim):
            if i < len(cell_count):
                rescell[i] = cell_count[i]
            else:
                rescell[i] = 0
        list_Betti.append(resbetti)
        list_simplex.append(rescell)
    return(np.array(list_Betti), np.array(list_simplex))

def add_random_direction(g):
    '''
    g in networkx graph
    returns: g with random directions
    '''
    new_edge_list = []
    for (u,v) in g.edges():
        r = random.randint(0, 2)
        if r == 0 or r == 1:
            new_edge_list.append((v,u))
        else:
            new_edge_list.append((u,v))
    h = nx.DiGraph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(new_edge_list)
    return h

def gen_ER(p_values,n):
    '''p_values: list of numbers between 0 and 1
    Generates an ER graph on n vertices with parameter p, for each p in p_values'''
    graphs = []
    i = 0
    print('Generating ER graphs')
    for p in p_values:
        print(i,end='\r')
        i += 1
        g = nx.gnp_random_graph(n,p,directed=True)
        graphs.append(g)
    return(graphs)

def gen_GR(r_values,n):
    '''r_values: list of numbers between 0 and 1
    Generates an GR graph on n vertices with parameter r, for each r in r_values'''
    graphs = []
    i = 0
    print('Generating GR graphs')
    for r in r_values:
        print(i,end='\r')
        i += 1
        gtemp = nx.random_geometric_graph(n,r)
        g = add_random_direction(gtemp)
        graphs.append(g)
    return(graphs)


def gen_PA(k_values,n):
    '''k_values: list of integers
    Generates an k-out with preferential attachment graph on n vertices with parameter k, for each k in k_values'''
    graphs = []
    i = 0
    print('Generating PA graphs')
    for k in k_values:
        print(i,end='\r')
        i += 1
        g = nx.random_k_out_graph(n,k,1,self_loops=False)
        graphs.append(g)
    return(graphs)


if __name__ == "__main__":
    # '''Example 1:
    # Generating graphs from different random models and their topological summaries'''
    # p_values = [random.uniform(0,0.1) for i in range(100)]
    # r_values =  [random.uniform(0,0.175) for i in range(100)]
    # k_values = [random.randint(4,30) for i in range(100)]
    # np.savetxt(path_graphs+'parameters',p_values+r_values+k_values)
    # n = 500
    # graphs = gen_ER(p_values,n) + gen_GR(r_values,n) + gen_PA(k_values,n)
    # nx.write_gpickle(graphs, path_graphs + 'all_graphs')

    '''Example 2:
    Generating approximate summaries, reading graphs from disc'''
    graphs = nx.read_gpickle(path_graphs + 'all_graphs')
    power = 10
    app = 10**power
    file_app = '_app_'+str(power)
    betti, simplex = top_summary(graphs,dim = 7,approx = app)
    np.savetxt(path_top + 'all_log_Betti' + file_app, np.log(np.maximum(betti, np.ones(betti.shape))))
    np.savetxt(path_top + 'all_log_simplex' + file_app, np.log(np.maximum(simplex, np.ones(simplex.shape))))

    