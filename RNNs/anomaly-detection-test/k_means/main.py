import math
import random
import csv, itertools
import pandas as pd
from constants import *
from helper_functions import *
from cluster import *
from k_means import *


plotly = False
try:
    import plotly
    from plotly.graph_objs import Scatter, Scatter3d, Layout
except ImportError:
    print "INFO: Plotly is not installed, plots will not be generated."



def main():

    max_len = 142000
    data_array = read_data_from_csv(file, COLUMNS, [], max_len)

    # For each of those points how many dimensions do they have?
    # Note: Plotting will only work in two or three dimensions

    # The K in k-means. How many clusters do we assume exist?
    num_clusters = 2

    # When do we say the optimization has 'converged' and stop updating clusters
    cutoff = 0.02

    fields_array = []
    fields_array.append('cpu_user_normal')
    fields_array.append('cpu_system_normal')
    fields_array.append('cpu_wait_normal')
    #fields_array.append('memory_used_normal')
    dimensions = len(fields_array)

    points = make_points_from_data(data_array, fields_array)


    # Cluster those data
    clusters = k_means(points, num_clusters, cutoff)

    #print out number of points in each cluster
    print ''
    print "printing number of points in each cluster"
    j = 1
    for c in clusters:
        percentage = float((100*(float(len(c.points)))))/float(max_len)
        print str(j) + ' '+str((len(c.points)))+'  %age:  '+str(percentage)
        j = j+1


    # for i, c in enumerate(clusters):
    #     for p in c.points:
    #         print " Cluster: ", i, "\t Point :", p

    # Display clusters using plotly for 2d data
    if dimensions in [2, 3] and plotly:
        print "Plotting points, launching browser ..."
        plotClusters(clusters, dimensions)



if __name__ == "__main__":
    main()
