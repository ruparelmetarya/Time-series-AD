from keras.models import Sequential, Model
from constants import *
from helper_functions import *
from keras.layers import Dense, Activation, Input
from keras import optimizers
from keras import losses
from keras.layers import LSTM
from keras.optimizers import RMSprop
import csv, itertools
import numpy as np
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



"""PARAMETERS"""
batch_size = 100
# series_length = 52000
input_size = 2998
features =  ['sample_count']
activation_fn ='relu'
loss_fn = 'mean_squared_error'
optimizer = optimizers.adagrad()
num_epochs = 10
num_iterations = 2




#takes in a dict and converts to array with normalized data
def make_points_from_data(data_array, params, start, end):
    array = []
    for param in params:
        print param
        list = []
        maxx = float(get_max(data_array, param, start, end))
        minn = float(get_min(data_array, param, start, end))
        for i in range(start, end):
            val = float(data_array[i][param])
            val = val -minn
            val = val/(maxx- minn)
            list.append(val)
        array.append(list)
    array = np.reshape((array), (1, len(params), input_size))
    return array


