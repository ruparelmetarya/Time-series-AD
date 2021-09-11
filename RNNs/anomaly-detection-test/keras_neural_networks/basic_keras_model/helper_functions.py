from keras.models import Sequential, Model
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
from constants import *



def plot_one_parameter(ylabel, y_array, dir_name ):
    figure = plt.figure()
    plt.ylabel(ylabel)
    plt.plot(y_array)
    plt.grid(True)
    plt.savefig(dir_name + ylabel+'.png')

def plot_two_parameters(xlabel, ylabel, x_array, y_array, grid, dir_name):
    figure = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x_array, y_array)
    plt.grid(grid)
    plt.savefig(dir_name + xlabel + ylabel + '.png')

def read_data_from_csv(file_name, field_names, data_array, max_len):
    with open(file_name, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in itertools.islice(reader, max_len):
            data_array.append(row)
    return data_array

def make_points_from_data(data_array, params, start, end):
    array = []
    for i in range(start, end):
        list = []
        for param in params:
            list.append(float(data_array[i][param]))
        array.append(list)
    array = np.reshape((array), (input_size, 1, 2))
    return array

