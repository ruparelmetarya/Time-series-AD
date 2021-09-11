import math
import random
import csv, itertools
import pandas as pd
from constants import *
from helper_functions import *
from cluster import *


def k_means(points, k, cutoff):

    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    # Note: Cluster takes lists, so we wrap each point in a list here.
    clusters = [Cluster([p]) for p in initial]

    # Loop through the data set until the clusters stabilize
    loop_counter = 0

    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for _ in clusters]
        cluster_count = len(clusters)

        # Start counting loops
        loop_counter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = get_distance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            cluster_index = 0

            # For the remainder of the clusters ...
            for i in range(cluster_count - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = get_distance(p, clusters[i + 1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i+1
            # After finding the cluster the smallest distance away
            # set the point to belong to that cluster
            lists[cluster_index].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # For each cluster ...
        for i in range(cluster_count):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # if(loop_counter>12):
        #     print "Converged after %s iterations" % loop_counter
        #     break

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff:
            print "Converged after %s iterations" % loop_counter
            break
    return clusters
