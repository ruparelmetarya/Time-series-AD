from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras import losses
from keras.layers import LSTM
from keras.optimizers import RMSprop
import csv, itertools
import numpy as np
import matplotlib as matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from constants import *
from parameters import *
from helper_functions import *



data_array = read_data_from_csv(file, COLUMNS, [], series_length)

input_array = np.array(make_points_from_data(data_array, 'cpu_user_normal', 1, 71000))

labels_array = np.array(make_points_from_data(data_array, 'cpu_user_normal', 71001, 142000))


print('Build model...')
funct_relu = 'relu'
funct_softmax = 'softmax'
funct_tanh = 'tanh'
model = Sequential()
model.add(Dense(batch_size, input_shape=(1,)))
model.add(Activation(funct_softmax))
model.add(Dense(output_dim=1))


optimizer = RMSprop(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()


for iteration in range(1, 5):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(input_array, labels_array, batch_size=batch_size, epochs=10, verbose=0)


loss = model.evaluate(input_array, labels_array, batch_size=batch_size, verbose = 0)
#print score
output = model.predict(input_array, batch_size, verbose=0)
# print 'max_input   '+ str(max(input_array))
print 'max predicted  ' +str(max(output))
print 'mean_input   '+str(np.mean(input_array))
print 'mean_predicted  '+str(np.mean(output))
print 'loss   '+ str(loss)
#plot_one_parameter('input', input_array, './plots/')
plot_one_parameter('output', output, './plots/')
plot_two_parameters('input', 'output', input_array, output, True,'./plots/')

