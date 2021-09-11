import math
import random
import csv, itertools
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1, x2)
print result
sess = tf.Session()
print(sess.run(result))
sess.close()

batch_size = 100

n_classes = 10
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    #(input_data* weight) + bias
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, 500])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #cycles feed forward plus back prop
    hm_epcochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epcochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y: epoch_y})
                epoch_loss +=c
            print ('Epoch ', epoch, 'completed out of ', hm_epcochs, ' loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x, y)


