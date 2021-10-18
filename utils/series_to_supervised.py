import pandas as pd


def series_to_supervised(data, columns=None, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        if columns is not None:
            names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
        else:
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            if columns is not None:
                names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
            else:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            if columns is not None:
                names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        clean_agg = agg.dropna()
    return clean_agg
    # return agg


if __name__ == '__main__':
    pd.set_option('max_colwidth', 400)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)

    dataframe = pd.read_csv('../data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset_columns = dataframe.columns
    # 将整型变为float
    dataset = dataset.astype('float32')

    reframed = series_to_supervised(dataset, dataset_columns, 3, 2, True)

    print(reframed.head())

    raw = pd.DataFrame()
    # columns = ['test%s' % (i + 1) for i in range(8)]
    raw["ob1"] = [x for x in range(10)]
    raw["ob2"] = [x for x in range(50, 60)]
    raw["ob3"] = [x for x in range(100, 110)]
    raw["ob4"] = [x for x in range(150, 160)]
    raw["ob5"] = [x for x in range(200, 210)]
    values = raw.values
    data = series_to_supervised(values, n_in=4, n_out=3)
    print(data.head())
