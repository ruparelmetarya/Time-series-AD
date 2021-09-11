from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
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
from helper_functions import *
from parameters import *



data_array = read_data_from_csv(file, COLUMNS, [], series_length)

input_array = np.array(make_points_from_data(data_array, ['cpu_user_normal', 'memory_used_normal'], 1, 71000))

labels_array = np.array(make_points_from_data(data_array, ['cpu_user_normal', 'memory_used_normal'], 71001, 142000))

print('Build model...')


# model = Sequential()
# model.add(LSTM(batch_size, input_shape= (2,)))
# model.add(LSTM(200, input_shape=(1,), activation='softmax'))
# model.add(LSTM(batch_size, input_shape= (1, 2), return_sequences=True))
# model.add(LSTM(batch_size, input_shape= (1, 2), return_sequences=True))

a = Input(shape = (1, 2))
predictions = Dense(2, activation='relu')(a)

model = Model(inputs = a, outputs=predictions)

optimizer = RMSprop(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()


for iteration in range(1, 5):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(input_array, labels_array, batch_size=200, epochs=10, verbose=0)


loss = model.evaluate(input_array, labels_array, batch_size=batch_size, verbose = 0)
#print score
output = model.predict(input_array, batch_size, verbose=0)
# print 'max_input   '+ str(max(input_array))
# print 'max predicted  ' +str(max(output))
# print 'mean_input   '+str(np.mean(input_array))
# print 'mean_predicted  '+str(np.mean(output))
print 'loss   '+ str(loss)
# plot_one_parameter('input', input_array, './plots/')
# plot_one_parameter('output', output, './plots/')
# plot_two_parameters('input', 'output', input_array, output, True,'./plots/')
