from rnn_model_parameters import *
from helper_functions import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from keras_diagram import ascii
import matplotlib as matplotlib
matplotlib.use('TkAgg')

#function to build lstm model
def build_lstm_model(batch_size=batch_size, features=features, input_size = input_size):
    model = Sequential()
    model.add(LSTM(batch_size, input_shape=(len(features), input_size), return_sequences=True))
    model.add(Dense(input_size))
    model.compile(loss=loss_fn, optimizer=optimizer)
    model.summary()
    return model


#GETTING THE DATA
data_array = read_data_from_csv(hbase_file, hbase_columns, [])
# print data_array

train_input = np.array(make_points_from_data(data_array, features, 1, input_size + 1))
# print "train:"
# print train_input
test_array = np.array(make_points_from_data(data_array, features, input_size + 2, 2 * (input_size + 1)))
# print "test"
# print test_array

#BUILDING THE MODEL
print('Build model...')
model = build_lstm_model()


#TENSORFLOW LOGS
tensorflow_visual = callbacks.TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=32, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


#FITTING THE MODEL
for iteration in range(1, num_iterations):
    model.fit(train_input, train_input, batch_size=batch_size, epochs=num_epochs, verbose=0, callbacks=[tensorflow_visual])


#EVALUATE THE LOSS
loss = model.evaluate(train_input, train_input, batch_size=batch_size, verbose = 0)
print 'loss   '+ str(loss) + '\n\n'


#MAKE THE PREDICTIONS
predictions = model.predict(train_input, batch_size, verbose=0)


#VISUALIZE THE MODEL
# plot_model(model, show_shapes=True, to_file='model.png')
# print(ascii(model))


#PLOT GRAPHS
plot_graphs(test_array, predictions, features, './plots/')

#PRINT MIN, MAX, AVG
get_min_max_avg(test_array, predictions, features)
