import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import scipy.io
import os
from Helper_Functions import *
import time
import random

#####################################################################################
#----------------------RAW HELPER FUNCTIONS-----------------------------------------#
#####################################################################################


def get_sim_data(name):
    sim_data = {}
    data = name.split('_') #Split into all parameters
    sim_data['num_particles'] = int(data[2][1:]) #Number of particles
    sim_data['Nr'] = int(data[4])  #Extract number ratio
    sim_data['packdens'] = float(data[6])
    sim_data['delV'] = float(data[8])
    sim_data['fluc'] = float(data[10])
    return sim_data

#-----------------------------------------------------------------
#*****************************************************************
#-----------------------------------------------------------------

def get_unique_params(files_path):

    file_names = os.listdir(files_path)
    file_params = list(map(get_sim_data, file_names))

    unique_params = {'num_particles':[],
        'Nr': [],
        'packdens': [],
        'delV': [],
        'fluc': [],  }
    
    for file_param in file_params:
        for key in file_param:
            if file_param[key] not in unique_params[key]:
                unique_params[key].append(file_param[key])

    print("\n UNIQUE PARAMETERS:")
    for key in unique_params:
        print(key, " : ", unique_params[key])
                
    return unique_params

#-----------------------------------------------------------------
#*****************************************************************
#-----------------------------------------------------------------

def parametric_sim_data(file_names, file_params, unique_params, fixed_params):
    
    for key in fixed_params:
        if fixed_params[key] not in unique_params[key]:
            raise Exception("There are no simulations with this parameter set")
            
    new_file_names = []
    
    for i in range(len(file_params)):
        flag = 1
        for key in fixed_params:
            if file_params[i][key] != fixed_params[key]:
                flag = 0
        if flag:

            new_file_names.append(file_names[i])
    
    return new_file_names

#-----------------------------------------------------------------
#*****************************************************************
#-----------------------------------------------------------------

def get_relative_positions(X_pos, Y_pos, index):
    rel_X = X_pos - np.reshape(X_pos.T[index], [X_pos.shape[0], -1])
    rel_Y = Y_pos - np.reshape(Y_pos.T[index], [Y_pos.shape[0], -1])
    #Experimental: Theoretically, the exact position of the particle should not be an indicator of its type, or even
    #play any helpful effect along with the relative positions, so we delete it from our inputs.
    rel_X = np.delete(rel_X, index, 1)
    rel_Y = np.delete(rel_Y, index, 1)  
    return rel_X, rel_Y