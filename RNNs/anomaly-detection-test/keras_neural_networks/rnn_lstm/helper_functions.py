from keras.models import Sequential, Model
from constants import *
from keras import callbacks
from rnn_model_parameters import *
from keras.layers import Dense, Activation, Input
from keras import optimizers
from keras import losses
from keras.layers import LSTM
from keras.optimizers import RMSprop
import csv, itertools
import pydot
import graphviz
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras_diagram import ascii
from keras.models import model_from_json
import numpy as np



class Graph:

    def __init__(self):
        self = self

    #plot univariate data
    def plot_one_parameter(self, ylabel, y_array, dir_name ):
        plt.ylabel(ylabel)
        plt.plot(y_array)
        plt.grid(True)
        plt.savefig(dir_name + ylabel+'.png')

    #plot two variables
    def plot_two_parameters(self, xlabel, ylabel, array, grid, dir_name, length, i, j):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(array[0][i], array[0][j])
        plt.grid(grid)
        plt.savefig(dir_name + xlabel + ylabel + '.png')
        plt.close()

#reads data from csv file
def read_data_from_csv(file_name, field_names, data_array):
    with open(file_name, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in reader:
            data_array.append(row)
    # print data_array
    return data_array

#return average of a list of numbers
def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

#returns the max value of a parameter from an array
def get_max(data_array, param, start, end):
    maxx = 0
    for i in range(start, end):
        try:
            val = float(data_array[i][param])
            if(val>maxx):
                maxx = val
        except IndexError:
            print 'error'
            print i
    return maxx

#returns the min value of a parameter from an array
def get_min(data_array, param, start, end):
    minn = 1000000
    for i in range(start, end):
        try:
            val = float(data_array[i][param])
            if (val < minn):
                minn = val
        except IndexError:
            print 'error'
            print i
    return minn

#load saved model from file
def load_model_from_file(file_name):
    json_file = open(file_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

#saves model to a json file
def save_model_to_json(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

#PRINT AVG, MIN, MAX for both actual and predicted sets
def get_min_max_avg(test_array, predictions, features):

    for i in range(len(features)):
        print str(features[i])+'\n'
        print 'Actual Max' + ' '+str(max(np.array(test_array[0][i])))
        print 'Predicted Max '  + ' ' + str(max(predictions[0][i])) +'\n'

        print 'Actual Min ' + ' in test set ' + str(min(np.array(test_array[0][i])))
        print 'Predicted Min ' + ' ' + str(min(np.array(predictions[0][i]))) + '\n'

        print 'Actual Avg ' + ' in test set ' + str(mean(np.array(test_array[0][i])))
        print 'Predicted Avg ' + ' ' + str(mean(np.array(predictions[0][i]))) + '\n\n'

#PLOT one-d and two-d graphs for all features
def plot_graphs(test_array, predictions, features, dir_name):
    #GRAPH PLOTTING..
    g1 = Graph()
    g3 = Graph()
    g5 = Graph()

    for i in range(len(features)):
        g1.plot_one_parameter('actual_ ' + str(features[i]), test_array[0][i], dir_name)
        g3.plot_one_parameter('predicted_ ' + str(features[i]), predictions[0][i], dir_name)
        plt.close()
        k = 0
        l =0
        if (i<(len(features)-1)):
            k = i
            l = k+1
        elif (i==(len(features)-1)):
            k = i
            l = 0
        #g5.plot_two_parameters_2('predicted_ ' + str(features[i]), '_ predicted_ '+str(features[1]), predictions, True,
        # './plots/', input_size, k, l)
        plt.close()

