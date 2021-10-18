from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

'''
下面的split_sequence（）函数实现了这种行为，并将给定的单变量序列分成多个样本，其中每个样本具有指定的时间步长，输出是单个时间步。
'''


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


if __name__ == '__main__':
    # define input sequence
    raw_seq = [i * 2 for i in range(1, 300)]
    print(raw_seq)
    # print(raw_seq[-3:])
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    print(X, y)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(n_steps, n_features)))  # 隐藏层，输入，特征维
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=70, batch_size=8, verbose=2)  # 迭代次数，批次数，verbose决定是否显示每次迭代
    # model.save_weights('../models/test.h5')
    # demonstrate prediction
    for i in range(10):
        x_input = array(raw_seq[-n_steps:])
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        print(x_input, yhat)
        print(yhat)
        raw_seq.append(yhat[0][0])
    print(raw_seq)

    plt.plot(raw_seq)
    plt.show()


