1. Data_paths.py contains all data paths that are used throughout to read or write information to disk. Variables(paths) can be modified to match user's needs.
1. model_dictionary.py contains a list of the models and parameters that we are using for our analysis.
1. 'Dataset{1,2}/Graphs/all_graphs' contains a list with all the random graphs that we generated.
1. 'Dataset{1,2}/TopSummaries/all_{log_Betti, log_simplex}' contains all the topological summaries in the same order as the list of graphs.
1. 'Dataset{1,2}/DistanceMatrices/' contains all the distance matrices in *.npy format.
1. Use distance_analysis.py to analyse the distance matrices of any subset of parameters listed in model_dictionary.py