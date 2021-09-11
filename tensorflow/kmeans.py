# from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

MAX_ITER = 100000
K = 5
N = 2000

# Reading the sample_counts from the CSV file and return it in "points"

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[1111111111111L], [1111111111111L], [""], [0], [0], [0], [0],[0]]
    start, stop, pod, sampleCount, threadCount1, threadCount2, threadCount3, threadCount4 = tf.decode_csv(csv_row, record_defaults=record_defaults)
    return sampleCount
filenames=["data (1).csv"]
filename_queue = tf.train.string_input_producer(filenames)
sampleCount = create_file_reader_ops(filename_queue)
point = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    i = 0
    while i < N :
        i = i + 1
        try :
            point += sess.run([sampleCount])
        except tf.errors.OutOfRangeError :
            break
# print(point)
sess= tf.Session()

# Reshaping the point to [30,2] tensor.
points = tf.reshape(point,[N/2,2])
# print(sess.run([points]))

# Assigning random centroids for all the points in "points"
centroids = tf.Variable(tf.slice(tf.random_shuffle(points),[0,0],[K,-1]))
init = tf.global_variables_initializer()
sess.run(init)
# print(sess.run([centroids]))

#Reshaping the points and centroids such that we can use that for the distances.
points_reshaped = tf.expand_dims(points,0)
centroids_reshaped = tf.expand_dims(centroids, 1)
# print(sess.run([centroids_reshaped]))

# The difference square and assigning the cluster
distances = tf.reduce_sum(tf.square(points_reshaped - centroids_reshaped), 2)
assignments = tf.argmin(distances, 0)
# print(sess.run([distances]))

# Now updating centroids with the Mean values of the currently assigned points in the Cluster
means = []
for c in xrange(K):
    means.append(tf.reduce_mean(tf.gather(points,tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),reduction_indices = [1]))
new_centroids = tf.concat(means,0)
update_centroids = tf.assign(centroids, new_centroids)
# print(sess.run([update_centroids]))



# Plotting the graph
with tf.Session() as sess:
  sess.run(init)
  # print(sess.run([distances]))
  for step in xrange(MAX_ITER):
    [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

print(points_values)
# print(assignment_values)
# print(centroid_values)
plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
