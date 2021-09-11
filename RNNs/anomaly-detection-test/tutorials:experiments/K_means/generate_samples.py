import tensorflow as tf
import numpy as np

from functions import create_samples
from functions import plot_clusters

n_features = 2
n_clusters = 2
n_samples_per_cluster = 50000
seed = 700
embiggen_factor = 70

# np.random.seed(seed)


import pandas as pd


"""Downloading data """
file = "../../data/cpu/CpuPerc.cpu.idle.csv"


"""Reading files into pandas dataframes"""
COLUMNS = ["system.PHX.SP1.na44:CpuPerc.cpu.idle{device=na44-app1-1-phx.ops.sfdc.net}"]


# readcsv returns a data frame
data_frame = pd.read_csv(file, names=COLUMNS, skipinitialspace=True)


centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

model = tf.global_variables_initializer()

with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)

plot_clusters(sample_values, centroid_values, n_samples_per_cluster)