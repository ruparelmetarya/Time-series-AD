from __future__ import print_function
import numpy as np
import tensorflow as tf
N = 20
K = 4
# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')
world = tf.constant('Hello SuckaSS!')
node_3 = tf.add(hello,world)
# Start tf session
sess = tf.Session()

# Run the op
print(sess.run([hello,world]))
print(sess.run([node_3]))

# points = tf.Variable(tf.random_uniform([N,2]))

# rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
# sess.run(init)
# print(sess.run([points]))
# print(sess.run([rep_points]))


points_n = 50
clusters_n = 3
iteration_n = 100

points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
# print(sess.run([points]))
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
init = tf.initialize_all_variables()
sess.run(init)
print(sess.run([centroids]))
points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)
print(sess.run([centroids_expanded]))
