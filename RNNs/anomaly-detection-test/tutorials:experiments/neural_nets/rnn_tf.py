from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn_cell
import csv, itertools



num_epochs = 100  # epoch is one set of data to be learnt..so, dividing up data into {num_epochs} parts


total_series_length = 142000  #number of data points


truncated_backprop_length = 15 #number of layers that we go back each time we back-propogate

state_size = 4 # n- size of state


num_classes = 4 # number of feature's we're looking at


echo_step = 3 # The output will be the echo of the input, shifted echo_step steps to the right

batch_size = 10 # looking at data in batches..this is the size of the batch..

num_batches = total_series_length//batch_size//truncated_backprop_length #number of batches


"""DOWNLOADING AND READING DATA"""
file = "../../../Desktop/CpuPerc.cpu.csv"

"""Reading files into pandas dataframes"""
COLUMNS = ["time_stamp", "cpu_user", "cpu_system", "cpu_wait", "memory_cached", "memory_free", "memory_used",
           "load_longterm", "load_midterm", "load_short_term", "cpu_user_normal", "cpu_system_normal",
           "cpu_wait_normal", "memory_cached_normal", "memory_free_normal", "memory_used_normal",
           "load_longterm_normal", "load_midterm_normal"]



def read_data_from_csv(file_name, field_names, data_array, max_len):
    with open(file_name, 'rb') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',', quotechar='|', fieldnames=field_names)
        for row in itertools.islice(reader, max_len):
            data_array.append(row)
    return data_array


def make_points_from_data(data, param, start, end):
    array = []
    for i in range(start, end):
        array.append(float(data[i][param]))
    return array


def generateData(data_array, start, end):
    x = np.array(make_points_from_data(data_array, "cpu_user_normal", start, end ))
    z = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))

    y = np.roll(z, echo_step)
    y[0:echo_step] = 0
    x = x.reshape(batch_size, 1) # The first index changing slowest, subseries as rows
    y = y.reshape(batch_size, -1)
    return (x, y)


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
init_state = tf.zeros([batch_size, state_size])


cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])





W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)  #weight for states
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)  #bias


# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1) #split tensor into subtensors
labels_series = tf.unstack(batchY_placeholder, axis=1)



# Forward passes
cell = rnn_cell.LSTMCell(state_size) #create LSTM cell
states_series, current_state = tf.nn.static_rnn(cell, inputs_series, initial_state=init_state, dtype = tf.float32)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

print ('hello')



losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)



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
dimensions = len(fields_array)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData(data_array, epoch_idx +1, epoch_idx+1+batch_size)

        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]
            batchX.reshape(batch_size, 1)
            print ('batch X shape   ' + str(batchX.shape) + 'batch X placeholder shape  ' + str(batchX_placeholder.shape))
            print ('batch Y shape    ' + str(batchY.shape) + 'batchY_placeholder shape   ' + str(batchY_placeholder.shape))
            print ('cell state_shape'+ str(cell_state.shape)+'  current cell state shape  '+str(_current_cell_state.shape))
            print ('hidden state shape    ' + str(hidden_state.shape) + 'current hidden state shape   ' + str(_current_hidden_state.shape))





            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX, batchY_placeholder: batchY, cell_state: _current_cell_state,  hidden_state: _current_hidden_state

                })

            _current_cell_state, _current_hidden_state = _current_state


            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
