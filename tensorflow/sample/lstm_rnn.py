import numpy
import matplotlib.pyplot as plt
import pandas
import math
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix.
# data_x_out has values at x(t)
# data_y_out has values at x(t+1)
# A mapping like :
# X | Y
# 1 | 2
# 2 | 4
# 4 | 3456
# 3456 | 6
def create_trainable_dataset(data_set, look_back=1):
    data_x_out, data_y_out = [], []
    for i in range(len(data_set) - look_back - 1):
        a = data_set[i:(i + look_back), 0]
        data_x_out.append(a)
        data_y_out.append(data_set[i + look_back, 0])
    return numpy.array(data_x_out), numpy.array(data_y_out)


numpy.random.seed(7)

# load the dataset
data = pandas.read_csv('todaydemo.csv', usecols=[3], engine='python')
data_set = data.values
data_set = data_set.astype('float32')
# print data_set

# normalize the dataset. As LSTM works best when the Value is b/w 0 & 1
# normalization is based on Scikit's Min Max Scaler
scaling = MinMaxScaler(feature_range=(0, 1))
data_set = scaling.fit_transform(data_set)
# print data_set

# split into train and test sets
# Splitting the csv data into
# 67% : Train , 33% : Test (The recommended split for almost all data)
train_length = int(len(data_set) * 0.67)
test_length = len(data_set) - train_length
train_data = data_set[0:train_length, :]
test_data = data_set[train_length:len(data_set), :]
# print(train_data)
# print(test_data)

look_back = 1
trainX, trainY = create_trainable_dataset(train_data, look_back)
print(trainX, "\n", trainY)
testX, testY = create_trainable_dataset(test_data, look_back)
# print trainX
# print trainY

# reshape input to be [samples, time steps, features]
# As the previous one was in the kind of form [samples, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# print trainX

# Keras Sequential LSTM Model
# The Model has a visible layer with 1 input
# a hidden layer with 4 LSTM neurons
# Ouput layer with single prediction values
# 100 epochs and 1 batch size
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# We invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data
trainPredict = scaling.inverse_transform(trainPredict)
trainY = scaling.inverse_transform([trainY])
testPredict = scaling.inverse_transform(testPredict)
testY = scaling.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

# GRAPH PLOTTING :
trainPredictPlot = numpy.empty_like(data_set)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(data_set)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(data_set) - 1, :] = testPredict

print(trainPredict)
print("\n")
print(testPredict)
print("\n")
predicted = []
for i in trainPredict:
    for j in i:
        predicted.append(j)

for i in testPredict:
    for j in i:
        predicted.append(j)

# print predicted
data2 = pandas.read_csv('todaydemo.csv', engine='python')

start_time_list = []
end_time_list = []
pod_list = []
samplecount_list = []
threads1_list = []
threads2_list = []
threads3_list = []
threads4_list = []
for i in range(len(data2['start_time'])):
    start_time_list.append(data2['start_time'][i])
    end_time_list.append(data2['end_time'][i])
    pod_list.append(data2['pod'][i])
    samplecount_list.append(data2['SampleCount'][i])
    threads1_list.append(data2['threadCount1'][i])
    threads2_list.append(data2['threadCount2'][i])
    threads3_list.append(data2['threadCount3'][i])
    threads4_list.append(data2['threadCount4'][i])

with open('todaypredicted1.csv', 'w') as csvfile:
    fieldnames = ['start_time', 'end_time', 'pod', 'SampleCount', 'threadCount1', 'threadCount2', 'threadCount3',
                  'threadCount4', 'predicted']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predicted)):
        writer.writerow({'start_time': start_time_list[i], 'end_time': end_time_list[i], 'pod': pod_list[i],
                         'SampleCount': samplecount_list[i],
                         'threadCount1': threads1_list[i], 'threadCount2': threads2_list[i],
                         'threadCount3': threads3_list[i], 'threadCount4': threads4_list[i], 'predicted': predicted[i]})

plt.plot(scaling.inverse_transform(data_set))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
