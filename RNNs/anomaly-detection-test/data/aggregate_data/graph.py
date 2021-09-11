import tensorflow as tf
import numpy as np
import csv
import tempfile
import datetime as dt
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from numpy import median
from random import choice, shuffle
from numpy import array
from mpl_toolkits.mplot3d import Axes3D as axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib.colors import colorConverter
from matplotlib.collections import PolyCollection

end = 142000

class Graph:
    def __init__(self):
        self = self

    #plot one parameter
    def plot_one_parameter(self, ylabel, y_array, dir_name):
        plt.ylabel(ylabel)
        plt.plot(y_array)
        plt.grid(True)
        plt.savefig(dir_name+'_ylabel.png')

    #plot two parameters
    def plot_two_parameters(self, xlabel, ylabel, x_array, y_array, grid):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x_array, y_array)
        plt.grid(grid)
        plt.savefig('plots/2-' + xlabel + ylabel + '.png')

    #plot three  parameters
    def plot_three_as_wireframe(self, xlabel, ylabel, zlabel, X, Y, Z):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = X
        Y = Y
        Z = Z
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        plt.savefig('plots/3-wireframe_' + xlabel + '_' + ylabel + '_' + zlabel + '.png')

    def plot_three_as_surface(self, xlabel, ylabel, zlabel, X, Y, Z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_zlabel(zlabel)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('plots/3-surface_' + xlabel + '_' + ylabel + '_' + zlabel + '.png')

    def plot_three_as_trisurf(self, xlabel, ylabel, zlabel, X, Y, Z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_zlabel(zlabel)
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('plots/3-trisurf_' + xlabel + '_' + ylabel + '_' + zlabel + '.png')


class HelperFunctions:

    def __init__(self):
        self = self

    def div_timestamp_by_thousand(self,rows, start, end):
        for i in range(start, end):
            row = rows[i]
            time_stamp = (int)(row['time_stamp'])
            time_stamp = time_stamp / 1000
            row['time_stamp'] = time_stamp
        return rows

    def divide_time_stamp_by_1000(self, row):
        timestamp = row['time_stamp']
        timestamp = timestamp / 1000
        row['time_stamp'] = timestamp
        return row

    def subtract_time_stamp_by_val(self, row, val):
        time_stamp = row['time_stamp']
        time_stamp = float(time_stamp) - val
        row['time_stamp'] = time_stamp
        return row

    #divides time_stamp by 1000 and sets starting value to 0
    def rescale_time_stamp(self, rows, start, end):

        first_row = rows[start]
        first_time_stamp = first_row['time_stamp']

        for i in range(start, end):
            row = rows[i]
            rows[i] = self.subtract_time_stamp_by_val(row, float(first_time_stamp))
            rows[i] = self.divide_time_stamp_by_1000(rows[i])

        return rows

    def normalize_data(self, array, start, end):
        avg = sum(array)
        lenn = len(array)
        avg = avg/lenn
        for i in range(start, end-1):
            try:
                val = array[i]
                val = val/avg
                array[i] = val
            except IndexError:
                print i
        return array


def read_data_from_csv(file_name, field_names, data_array):
    with open(file_name, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in reader:
            data_array.append(row)
    return data_array


def create_individual_array_from_aggregate_data(array, start, end, parameter):
    parameter_array = []
    for i in range(start, end-1):
        row = array[i]
        try:
            if(row[parameter]!='#N/A'):
                parameter_array.append(float(row[parameter]))
        except ValueError:
            print parameter
            print row[parameter]

    return parameter_array


COLUMNS = ["timestamp", "cpu-perc"]
file_name = "../../../../../Desktop/CpuPerc.cpu.csv"
data_frame = pd.read_csv(file_name, names=COLUMNS, skipinitialspace=True, na_values=[])
data_frame.fillna('', inplace=True)


rows = []

field_names = ['time_stamp', 'cpu_user', 'cpu_system', 'cpu_wait', 'memory_cached', 'memory_free', 'memory_used',
               'load_longterm', 'load_midterm', 'load_shortterm']


rows = read_data_from_csv(file_name, field_names, rows)
h = HelperFunctions()
rows = h.rescale_time_stamp(rows, 1, end)



#creating arrays for graph plotting
cpu_user = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'cpu_user'), 0, end)
cpu_system = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'cpu_system'), 0, end)
cpu_wait = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'cpu_wait'), 0, end)

memory_cached = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'memory_cached'),0, end)
memory_free = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'memory_free'), 0, end)
memory_used = h. normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'memory_used'), 0, end)
load_longterm = h.normalize_data(create_individual_array_from_aggregate_data(rows, 1, end, 'load_longterm'), 0, end)

g = Graph()

# # g.plot_two_parameters( 'memory_cached', 'cpu_user',  memory_cached, cpu_user,True)
# # g.plot_two_parameters('memory_used','cpu_user', memory_used, cpu_user, True)
# # g.plot_two_parameters('load_longterm', 'cpu_user',load_longterm,  cpu_user,  True)
# g.plot_two_parameters('memory_used','cpu_wait', memory_used, cpu_wait, True)
#
#
# # g.plot_three_as_trisurf('cpu_user', 'memory_used', 'load_longterm',cpu_user, memory_used, load_longterm)
# # g.plot_three_as_surface('cpu_user', 'memory_used', 'load_longterm',cpu_user, memory_used, load_longterm)
# # g.plot_three_as_wireframe('cpu_user', 'memory_used', 'load_longterm',cpu_user, memory_used, load_longterm)
# g.plot_three_as_wireframe('cpu_user', 'cpu_wait', 'memory_used', cpu_user, cpu_wait, memory_used)



g.plot_two_parameters( 'memory_cached', 'cpu_wait',  memory_cached, cpu_wait,True)
g.plot_two_parameters('memory_used','cpu_wait', memory_used, cpu_wait, True)
g.plot_two_parameters('load_longterm', 'cpu_wait',load_longterm,  cpu_wait,  True)
g.plot_two_parameters('memory_used','cpu_wait', memory_used, cpu_wait, True)


g.plot_three_as_trisurf('cpu_wait', 'memory_used', 'load_longterm',cpu_wait, memory_used, load_longterm)
g.plot_three_as_wireframe('cpu_wait', 'memory_used', 'load_longterm',cpu_wait, memory_used, load_longterm)
# g.plot_three_as_wireframe('cpu_system', 'cpu_wait', 'memory_used', cpu_system, cpu_wait, memory_used)