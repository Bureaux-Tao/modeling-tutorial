import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SF Mono']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 360  # 图片像素
plt.rcParams['figure.dpi'] = 360  # 分辨率


class GrayForecast():
    # 初始化
    def __init__(self, data, datacolumn=None):
        '''
        作为初始化的方法，我们希望它能将数据格式化存储，
        并且可使用的类型越多越好，在这里我先实现能处理三种类型：
        一维列表、DataFrame、Series。
        如果处理DataFrame可能会出现不止一维的情况，
        于是设定一个参数datacolumn，
        用于处理传入DataFrame不止一列数据到底用哪个的问题
        :param data: DataFrame    数据
        :param datacolumn: string       数据的含义
        '''

        if isinstance(data, pd.core.frame.DataFrame):
            self.data = data
            try:
                self.data.columns = ['数据']
            except:
                if not datacolumn:
                    raise Exception('您传入的dataframe不止一列')
                else:
                    self.data = pd.DataFrame(data[datacolumn])
                    self.data.columns = ['数据']
        elif isinstance(data, pd.core.series.Series):
            self.data = pd.DataFrame(data, columns=['数据'])
        else:
            self.data = pd.DataFrame(data, columns=['数据'])

        self.forecast_list = self.data.copy()

        if datacolumn:
            self.datacolumn = datacolumn
        else:
            self.datacolumn = None

    # save arg:
    #        data                DataFrame    数据
    #        forecast_list       DataFrame    预测序列
    #        datacolumn          string       数据的含义

    # 级比校验
    def level_check(self):
        '''
        按照级比校验的步骤进行，最终返回是否成功的bool类型值
        :return:
        '''
        # 数据级比校验
        n = len(self.data)
        lambda_k = np.zeros(n - 1)
        for i in range(n - 1):
            lambda_k[i] = self.data.ix[i]["数据"] / self.data.ix[i + 1]["数据"]
            if lambda_k[i] < np.exp(-2 / (n + 1)) or lambda_k[i] > np.exp(2 / (n + 2)):
                flag = False
        else:
            flag = True
        self.lambda_k = lambda_k
        if not flag:
            print("级比校验失败，请对X(0)做平移变换")
            return False
        else:
            print("级比校验成功，请继续")
            return True

    # save arg:
    #        lambda_k            1-d list

    # GM(1,1)建模
    def GM_11_build_model(self, forecast=5):
        '''
        按照GM(1,1)的步骤进行一次预测并增长预测序列（forecast_list）
        传入的参数forecast为使用forecast_list末尾数据的数量，
        因为灰色预测为短期预测，过多的数据反而会导致数据精准度变差
        :param forecast:
        :return:
        '''
        if forecast > len(self.data):
            raise Exception('您的数据行不够')
        X_0 = np.array(self.forecast_list['数据'].tail(forecast))
        # 1-AGO
        X_1 = np.zeros(X_0.shape)
        for i in range(X_0.shape[0]):
            X_1[i] = np.sum(X_0[0:i + 1])
        # 紧邻均值生成序列
        Z_1 = np.zeros(X_1.shape[0] - 1)
        for i in range(1, X_1.shape[0]):
            Z_1[i - 1] = -0.5 * (X_1[i] + X_1[i - 1])

        B = np.append(np.array(np.mat(Z_1).T), np.ones(Z_1.shape).reshape((Z_1.shape[0], 1)), axis=1)
        Yn = X_0[1:].reshape((X_0[1:].shape[0], 1))

        B = np.mat(B)
        Yn = np.mat(Yn)
        a_ = (B.T * B) ** -1 * B.T * Yn

        a, b = np.array(a_.T)[0]

        X_ = np.zeros(X_0.shape[0])

        def f(k):
            return (X_0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (k))

        self.forecast_list.loc[len(self.forecast_list)] = f(X_.shape[0])

    # 预测
    def forecast(self, time=5, forecast_data_len=5):
        '''
        预测函数只要调用GM_11_build_model就可以，
        传入的参数time为向后预测的次数，
        forecast_data_len为每次预测所用末尾数据的条目数
        :param time:
        :param forecast_data_len:
        :return:
        '''
        for i in range(time):
            self.GM_11_build_model(forecast=forecast_data_len)

    # 打印日志
    def log(self):
        '''
        打印当前预测序列
        :return:
        '''
        res = self.forecast_list.copy()
        if self.datacolumn:
            res.columns = [self.datacolumn]
        return res

    # 重置
    def reset(self):
        '''
        初始化序列
        :return:
        '''
        self.forecast_list = self.data.copy()

    # 作图
    def plot(self):
        '''
        作图
        :return:
        '''
        self.forecast_list.plot()
        if self.datacolumn:
            plt.ylabel(self.datacolumn)
            plt.legend([self.datacolumn])
