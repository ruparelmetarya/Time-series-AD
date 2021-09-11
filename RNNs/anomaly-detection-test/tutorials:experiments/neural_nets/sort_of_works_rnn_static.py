#IMPORTS
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell
import csv, itertools


#GLOBAL VARIABLES
truncated_backprop_length = 1 # number of truncated backprop steps
batch_size = 200

num_classes = 1

state_size = 4
total_series_length = 141000  #number of data points

learning_rate = 0.1

num_batches = total_series_length//batch_size//truncated_backprop_length #number of batches


"""DOWNLOADING AND READING DATA"""
file = "../../../Desktop/CpuPerc.cpu.csv"

"""Reading files into pandas dataframes"""
COLUMNS = ["time_stamp", "cpu_user", "cpu_system", "cpu_wait", "memory_cached", "memory_free", "memory_used",
           "load_longterm", "load_midterm", "load_short_term", "cpu_user_normal", "cpu_system_normal",
           "cpu_wait_normal", "memory_cached_normal", "memory_free_normal", "memory_used_normal",
           "load_longterm_normal", "load_midterm_normal"]

max_len = 142000

def read_data_from_csv(file_name, field_names, data_array, max_len):
    with open(file_name, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in itertools.islice(reader, max_len):
            data_array.append(row)
    return data_array

data_array = read_data_from_csv(file, COLUMNS, [], max_len)

def make_points_from_data(data, param, start, end):
    array = []
    for i in range(start, end):
        array.append(float(data[i][param]))
    return array

def gen_data(data_array, param, start, end):
    x = np.array(make_points_from_data(data_array, param, start, end))
    y = np.array(make_points_from_data(data_array, param, start+batch_size, end+batch_size))
    x = np.reshape(x, (batch_size, -1))
    y = np.reshape(y, (batch_size, -1))
    return x, y


def gen_batch(batch_number, data_array=data_array, param='cpu_user_normal'):
    start = (batch_number-1)*batch_size + 1
    end = start + batch_size
    x, y = gen_data(data_array, param, start, end)
    return x,y



"""
Placeholders
"""


x = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name='labels_placeholder')

init_state = tf.zeros([batch_size, state_size])

# x_ones = tf.ones([batch_size, num_classes])
# rnn_inputs = tf.unstack(x_ones, axis=1)
#
# x_one_hot = tf.one_hot(x, num_classes)
# rnn_inputs = tf.unstack(x_one_hot, axis=1)

rnn_inputs = []

for i in range(truncated_backprop_length):
    x_ones = tf.ones([batch_size, num_classes])
    rnn_inputs.append(x_ones)


y_list = []

for j in range(truncated_backprop_length):
    y_ones = tf.ones([batch_size, num_classes])
    y_list.append(y_ones)


"""
Definition of rnn_cell

This is very similar to the __call__ method on Tensorflow's BasicRNNCell. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py#L95
"""


with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

"""
Adding rnn_cells to graph

This is a simplified version of the "static_rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: In practice, using "dynamic_rnn" is a better choice that the "static_rnn":
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
"""
state = init_state
rnn_outputs = []


for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)


final_state = rnn_outputs[-1]


"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
function from Tensorflow's API, except that here we are using a list of 2D tensors, instead of a 3D tensor. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""

#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]

# Turn our y placeholder into a list of labels
y_as_list = y_list

#losses and train_step
losses = [tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
          logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


"""
Train the network
"""

def train_network(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        b = 1
        for epoch_idx in range(num_epochs):
            X, Y = gen_batch(b)
            b = b+1
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEPOCH", epoch_idx)

            tr_losses, training_loss_, training_state, _ = \
                sess.run([losses,
                          total_loss,
                          final_state,
                          train_step],
                              feed_dict={x:X, y:Y, init_state:training_state})
            training_loss += training_loss_
            print ('training losss  '+str(training_loss))
            if verbose:
                print("Average loss at step",
                      "for last 250 steps:", training_loss/100)
            training_losses.append(training_loss/100)
            training_loss = 0

    return training_losses


training_losses = train_network(10,truncated_backprop_length)
plt.plot(training_losses)
