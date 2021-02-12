

''' For Dataset1:
This is list of models and parameters where model with parameter 0.p is represented
by 'modelp'
'''
dataset = 'Dataset'
metrics_dict = ['PD', 'TEuc', 'TEMD','param', 'random']
feat_dict = ['top_log_Betti','top_log_simplex']
model_dict = ['ER03', 'ER06', 'ER1', 'ER15', 'ER2', 'ER25', 'GR1', 'GR175', 'GR3', 'PA20', 'PA40','PA70'] 
model_ER = ['ER03', 'ER06', 'ER1', 'ER15', 'ER2', 'ER25']
model_GR = ['GR1', 'GR175', 'GR3']
model_PA = [ 'PA20', 'PA40','PA70']
graphs_per_param = 10 # number of graphs for fixed parameter/model

# WARNING: if adding new models/parameters, add them at the end of this list - we are using a list and not a dictionary to obtain a consistent ordering used in iterations, in case more graphs are added.
# This allows us to append distances, rather than regenerate the whole distance list.



# '''For Dataset2:'''
# dataset = 'Dataset2'
# metrics_dict = ['PD', 'TEuc','TEMD','param','top_log_Betti_app_0','top_log_Betti_app_1','top_log_Betti_app_2','top_log_Betti_app_3','top_log_Betti_app_4', 'random']
# feat_dict = ['top_log_Betti','top_log_simplex']
# model_dict = ['ER','GR','PA']
# model_ER = ['ER']
# model_GR = ['GR']
# model_PA = ['PA']
# graphs_per_param = 100
