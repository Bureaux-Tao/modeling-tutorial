import copy

import numpy
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load the dataset
dataframe = read_csv('../../data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
# 将整型变为float
dataset = dataset.astype('float32')

data = copy.deepcopy(dataset)

numpy.random.seed(7)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back + 1, 0])
    return numpy.array(dataX), numpy.array(dataY)


# dataset = copy.deepcopy(data)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# use this function to prepare the train and test datasets for modeling
look_back = 10
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(8, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.load_weights('../../models/international-airline-passengers.h5')

for i in range(20):
    tmp = copy.deepcopy(data)
    tmp = scaler.fit_transform(tmp)
    newX, newY = create_dataset(tmp, look_back)
    newX = numpy.reshape(newX, (newX.shape[0], 1, newX.shape[1]))
    # print()
    newPredict = model.predict(np.array(newX[-look_back:]))
    newPredict = scaler.inverse_transform(newPredict)
    print(newPredict[0])
    data = np.append(data, newPredict[0])
    # print(data)
    data = data.reshape(-1, 1)

print(data)
plt.plot(data)
plt.show()
