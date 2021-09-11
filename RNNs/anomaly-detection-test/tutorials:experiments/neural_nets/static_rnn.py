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

num_batches = total_series_length//batch_size//truncated_backprop_length  #number of batches


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


"""GENERATING BATCHES"""
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
    return tf.stack(x), tf.stack(y)


"""
Placeholders
"""

x = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length], name='input_placeholder')
y = tf.placeholder(tf.float64, [batch_size, truncated_backprop_length], name='labels_placeholder')
init_state = tf.zeros([batch_size, state_size])

n_hidden = 512

lstm = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]



estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type= 2, prediction_type=2, sequence_feature_columns = features, num_units=512, cell_type='lstm', optimizer='Adam')
print ('estimator' +str(estimator))


# Define the training inputs
def get_train_inputs():
    x, y = gen_batch(1)
    return x

def get_test_inputs():
    x, y = gen_batch(1)
    return y


  # Fit model.
estimator.fit(input_fn=get_train_inputs, steps=2000)

# Evaluate accuracy.
accuracy_score = estimator.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


"""
Adding rnn_cells to graph

This is a simplified version of the "static_rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: In practice, using "dynamic_rnn" is a better choice that the "static_rnn":
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
"""
state = init_state

"""
Predictions, loss, training step

Losses is similar to the "sequence_loss"
function from Tensorflow's API, except that here we are using a list of 2D tensors, instead of a 3D tensor. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/loss.py#L30
"""

#rnn cell defined as:
def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)

def loss_func(actual_values, predicted_values):
    loss = tf.reduce_sum(tf.square(predicted_values-actual_values))
    return loss



"""
Adding rnn_cells to graph

This is a simplified version of the "static_rnn" function from Tensorflow's api. See:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py#L41
Note: In practice, using "dynamic_rnn" is a better choice that the "static_rnn":
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py#L390
"""
rnn_inputs = []
state = init_state
rnn_outputs = []

for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)
    rnn_outputs.append(state)


#logits and predictions
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]

predictions = [tf.nn.softmax(logit) for logit in logits]



#losses and train_step
total_loss = loss_func(y, logits)
optimizer = tf.train.GradientDescentOptimizer(0.01)
print ("total loss    " + str(total_loss))


def train_batches():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.initialize_all_variables())
        num_batches = total_series_length//batch_size
        training_state = tf.stack(np.zeros((batch_size, state_size)))
        for i in range(num_batches):
            batch_array, actual_values = gen_batch(i+1)
            predictions, values, training_state = sess.run({x:batch_array, y:actual_values, init_state:training_state})
        print ('all done')

train_batches()


"""
1. Generate data in batches
2. train data
"""