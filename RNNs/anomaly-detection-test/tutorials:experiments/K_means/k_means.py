# import tensorflow as tf
# import numpy as np
# import tempfile
# import matplotlib as matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import pandas as pd
# from random import choice, shuffle
# from numpy import np
# import matplotlib as plt
#
# MAX_ITERATIONS  = 1000
#
# # Function: K Means
# # -------------
# # K-Means is an algorithm that takes in a dataset and a constant
# # k and returns k centroids (which define clusters of data in the
# # dataset which are similar to one another).
# def kmeans(dataSet, k):
#     # Initialize centroids randomly
#     centroids = getRandomCentroids(dataSet, k)
#
#     # Initialize book keeping vars.
#     iterations = 0
#     old_centroids = None
#
#     # Run the main k-means algorithm
#     while not shouldStop(old_centroids, centroids, iterations):
#         # Save old centroids for convergence test. Book keeping.
#         old_centroids = centroids
#         iterations += 1
#
#         # Assign labels to each datapoint based on centroids
#         labels = get_labels(dataSet, centroids)
#
#         # Assign centroids based on datapoint labels
#         centroids = get_centroids(dataSet, labels, k)
#
#     # We can get the labels too by calling getLabels(dataSet, centroids)
#     return centroids
#
#
#
# # Function: Should Stop
# # -------------
# # Returns True or False if k-means is done. K-means terminates either
# # because it has run a maximum number of iterations OR the centroids
# # stop changing.
# def shouldStop(old_centroids, centroids, iterations):
#     if iterations > MAX_ITERATIONS: return True
#     return old_centroids == centroids
#
# def getRandomCentroids(points, k):
#     """returns k centroids from the initial points"""
#     centroids = points.copy()
#     np.random.shuffle(centroids)
#     return centroids[:k]
#
#
#
# # Function: Get Labels
# # -------------
# # Returns a label for each piece of data in the dataset.
# def get_labels(dataSet, centroids):
#     print 'hello'
#     # For each element in the dataset, chose the closest centroid.
#     # Make that centroid the element's label.
#
#     # Function: Get Centroids
#     # -------------
#     # Returns k random centroids, each of dimension n.
#
# def get_centroids(dataSet, labels, k):
#
#     print 'hello'
# # Each centroid is the geometric mean of the points that
# # have that centroid's label. Important: If a centroid is empty (no points have
# # that centroid's label) you should randomly re-initialize it.
#
#
# x = np.array([1, 2, 3, 4, 5]
# y = np.array([8, 8, 8, 8, 8])
# z = np.ones((5, 9))
#
# np.sqrt(sum((x - y) ** 2))
#
# np.sqrt(((z-x)**2).sum(axis=0))
#
#
#
# # start with one file from cpu folder
#
# """DOWNLOADING AND READING DATA"""
#
# file = "../../../Desktop/CpuPerc.cpu.csv"
#
# """Reading files into pandas dataframes"""
#
# COLUMNS = ["time_stamp", "cpu_user", "cpu_system", "cpu_wait", "memory_cached", "memory_free", "memory_used",
#            "load_longterm", "load_midterm"]
# CONTINUOUS_COLUMNS = ["cpu_user", "cpu_system", "cpu_wait", "memory_cached", "memory_free", "memory_used",
#            "load_longterm", "load_midterm"]
#
# # readcsv returns a data frame
# data_frame.fillna('', inplace=True)
#
#
# """CONVERTING DATA INTO TENSORFLOW OBJECTS"""
#
# continuous_cols = {k: tf.constant(data_frame[k].values) #tf.constant creates a tensor: https://www.tensorflow.org/api_docs/python/tf/constant
#                        for k in CONTINUOUS_COLUMNS}
#
# feature_cols = dict(continuous_cols.items());
#
# cpu = tf.contrib.layers.real_valued_column("cpu-perc")
# timestamp = tf.contrib.layers.real_valued_column("timestamp")
#
#
# model_dir = tempfile.mkdtemp()
# feature_columns = [cpu, timestamp]
#
#
#
# def input_fn(df):
#     # Creates a dictionary mapping from each continuous feature column name (k) to
#     # the values of that column stored in a constant Tensor.
#     continuous_cols = {k: tf.constant(df[k].values.astype(np.float32)) #tf.constant creates a tensor: https://www.tensorflow.org/api_docs/python/tf/constant
#                        for k in CONTINUOUS_COLUMNS}
#
#     # Merges the two dictionaries into one.
#     feature_cols = dict(continuous_cols.items()) #returns list of tuple pairs
#     # Converts the label column into a constant Tensor.
#     label = tf.constant(df[CONTINUOUS_COLUMNS].values.astype(np.float32))
#     # Returns the feature columns and the label.
#     return feature_cols, None
#
#
# def wrap_input_func():
#     return input_fn(data_frame)
#
# def TFKMeansCluster(vectors, num_clusters):
#
#     num_clusters = int(num_clusters)
#     print vectors
#
#     #assert num_clusters < len(vectors) #check if num_clusters less than the number of values in the vector
#
#     # Find out the dimensionality
#     dim = len(vectors[0])
#
#     # Will help select random centroids from among the available vectors
#     vector_indices = list(range(len(vectors)))
#     print vector_indices
#     shuffle(vector_indices) #random shuffle
#
#     # GRAPH OF COMPUTATION
#     # We initialize a new graph and set it as the default during each run
#     # of this algorithm. This ensures that as this function is called
#     # multiple times, the default graph doesn't keep getting crowded with
#     # unused ops and Variables from previous function calls.
#
#     graph = tf.Graph()
#
#
#     with graph.as_default():
#
#         # SESSION OF COMPUTATION
#
#         sess = tf.Session()
#
#         ##CONSTRUCTING THE ELEMENTS OF COMPUTATION
#
#         """First lets ensure we have a Variable vector for each centroid,
#         initialized to one of the vectors from the available data points"""
#
#         centroids = [tf.Variable(vectors[i], name='centroid')
#                      for i in range(0,1)]
#
#
#         """These nodes will assign the centroid Variables the appropriate values"""
#         centroid_value = tf.placeholder("float64", [dim])
#         cent_assigns = []
#         for centroid in centroids:
#             cent_assigns.append(tf.assign(centroid, centroid_value))
#
#             #
#
#         """Variables for cluster assignments of individual vectors(initialized to 0 at first)"""
#         assignments = [tf.Variable(0) for i in range(len(vectors))]
#         #
#
#         """These nodes will assign an assignment Variable the appropriate value"""
#         assignment_value = tf.placeholder("int32")
#         cluster_assigns = []
#         for assignment in assignments:
#             cluster_assigns.append(tf.assign(assignment,
#                                              assignment_value))
#
#         """Now lets construct the node that will compute the mean. The placeholder for the input"""
#         mean_input = tf.placeholder("float", [None, dim])
#
#         # The Node/op takes the input and computes a mean along the 0th
#         # dimension, i.e. the list of input vectors
#         mean_op = tf.reduce_mean(mean_input, 0)
#
#         """Node for computing Euclidean distances Placeholders for input"""
#         v1 = tf.placeholder("float", [dim])
#         v2 = tf.placeholder("float", [dim])
#         euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
#             v1, v2), 2)))
#
#         """This node will figure out which cluster to assign a vector to,
#         based on Euclidean distances of the vector from the centroids."""
#         # Placeholder for input
#         centroid_distances = tf.placeholder("float", [num_clusters])
#         cluster_assignment = tf.argmin(centroid_distances, 0)
#
#         """INITIALIZING STATE VARIABLES
#
#         This will help initialization of all Variables defined with respect
#         to the graph. The Variable-initializer should be defined after
#         all the Variables have been constructed, so that each of them
#         will be included in the initialization."""
#         init_op = tf.initialize_all_variables()
#
#         # Initialize all variables
#         sess.run(init_op)
#
#         """CLUSTERING ITERATIONS
#
#         Now perform the Expectation-Maximization steps of K-Means clustering
#         iterations. To keep things simple, we will only do a set number of
#         iterations, instead of using a Stopping Criterion."""
#         noofiterations = 100
#         for iteration_n in range(noofiterations):
#
#             """EXPECTATION STEP
#             Based on the centroid locations till last iteration,
#             compute the _expected_ centroid assignments.
#              Iterate over each vector"""
#
#             for vector_n in range(len(vectors)):
#                 vect = vectors[vector_n]
#
#                 # Compute Euclidean distance between this vector and each
#                 # centroid. Remember that this list cannot be named
#                 # 'centroid_distances', since that is the input to the
#                 # cluster assignment node.
#
#                 distances = [sess.run(euclid_dist, feed_dict={
#                     v1: vect, v2: sess.run(centroid)})
#                              for centroid in centroids]
#
#                 # Now use the cluster assignment node, with the distances
#                 # as the input
#                 assignment = sess.run(cluster_assignment, feed_dict={
#                     centroid_distances: distances})
#
#                 # Now assign the value to the appropriate state variable
#                 sess.run(cluster_assigns[vector_n], feed_dict={
#                     assignment_value: assignment})
#                 #
#
#             """MAXIMIZATION STEP
#             # Based on the expected state computed from the Expectation Step,
#             # compute the locations of the centroids so as to maximize the
#             # overall objective of minimizing within-cluster Sum-of-Squares"""
#
#             for cluster_n in range(num_clusters):
#                 # Collect all the vectors assigned to this cluster
#                 assigned_vects = [vectors[i] for i in range(len(vectors))
#                                   if sess.run(assignments[i]) == cluster_n]
#                 # Compute new centroid location
#                 new_location = sess.run(mean_op, feed_dict={
#                     mean_input: array(assigned_vects)})
#
#                 # Assign value to appropriate variable
#                 sess.run(cent_assigns[cluster_n], feed_dict={
#                     centroid_value: new_location})
#
#         # Return centroids and assignments
#         centroids = sess.run(centroids)
#         assignments = sess.run(assignments)
#         plot_clusters(centroids, assignments,50000)
#         return centroids, assignments
#
# def plot_clusters(centroids, all_samples, n_samples_per_cluster):
#     # Plot out the different clusters
#     # Choose a different colour for each cluster
#     colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
#
#     for i, centroid in enumerate(centroids):
#         # Grab just the samples fpr the given cluster and plot them out with a new colour
#         samples = all_samples[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster]
#         plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
#         # Also plot centroid
#         plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
#         plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
#     plt.show()
#
#
# print 'hello'
# print feature_columns
#
# #
# m = tf.contrib.learn.KMeansClustering(2, model_dir=model_dir)
# results = m.fit(input_fn=wrap_input_func, steps = 2)
#
#
# print 'bye'
#
#
# #feature_columns = input_fn(data_frame)
# #TFKMeansCluster(feature_columns, 2)
# #TFKMeansCluster(continuous_cols, 2)
#
#
#
