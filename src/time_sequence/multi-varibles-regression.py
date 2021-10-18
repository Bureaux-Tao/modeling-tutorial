# load data
import math
import os
import sys

sys.path.insert(0,'/Users/Bureaux/Documents/workspace/PyCharmProjects/Modeling')
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Sequential, layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.series_to_supervised import series_to_supervised

plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# 采用LSTM模型时，第一步需要对数据进行适配处理，
# 其中包括将数据集转化为有监督学习问题和归一化变量（包括输入和输出值），
# 使其能够实现通过前一个时刻（t-1）的污染数据和天气条件预测当前时刻（t）的污染。

# 以上的处理方式很直接也比较简单，仅仅只是为了抛砖引玉，其他的处理方式也可以探索，比如：
# 　　1. 利用过去24小时的污染数据和天气条件预测当前时刻的污染；
# 　　2. 预测下一个时刻（t+1）可能的天气条件；


if __name__ == '__main__':
    # os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # print(os.path.abspath(os.path.join(os.getcwd())))

    dataset = pd.read_csv('../../data/PRSA_data_2010.1.1-2014.12.31.csv', parse_dates=[['year', 'month', 'day', 'hour']],
                          index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)

    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'

    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours

    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))

    # save to file
    # dataset.to_csv('pollution.csv')

    values = dataset.values
    columns = dataset.columns
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()

    # integer encode direction
    # transform: ["male", "female", "female", "male"] -> [1 0 0 1]
    # inverse_transform: [1,0,0,1] -> ['male', 'female', 'female', 'male']
    print(values.shape)
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])

    # ensure all data is float
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    # series_to_supervised()函数将数据集转化为有监督学习问题
    reframed = series_to_supervised(scaled, columns, 1, 1)

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print()
    pd.set_option('max_colwidth', 400)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    print(reframed.head())

    # 将训练集和测试集划分为输入和输出变量，最终将输入（X）改造为LSTM的输入格式，即[samples, timesteps, features]。
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 2 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(
        layers.LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # fit network
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)  # 提前结束
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    history = model.fit(train_X, train_y, epochs=100, batch_size=16, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, callbacks=[early_stopping, reduce_lr])

    model.save_weights('../../models/multi_variables_regression.h5')
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    raw = inv_y.size
    inv_y = inv_y[-24 * 3:]
    inv_yHat = inv_yhat[-24 * 3:]
    plt.plot(inv_yHat, label='forecast')
    plt.plot(inv_y, label='observation')
    plt.ylabel('pm2.5')
    plt.legend()
    plt.show()
