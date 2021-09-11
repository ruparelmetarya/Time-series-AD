import tensorflow as tf
import numpy as np
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


"""Downloading data """
file = "../../data/cpu/CpuPerc.cpu.idle.csv"

"""Reading files into pandas dataframes"""
COLUMNS = ["system.PHX.SP1.na44:CpuPerc.cpu.idle{device=na44-app1-1-phx.ops.sfdc.net}"]


# readcsv returns a data frame
data_frame = pd.read_csv(file, names=COLUMNS, skipinitialspace=True)


def choose_random_centroids(samples, n_clusters):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids



def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []

    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))

        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    print samples
    centroids = tf.concat(centroids, 0, name='centroids')
    print centroids
    return centroids, samples




def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster]
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()








