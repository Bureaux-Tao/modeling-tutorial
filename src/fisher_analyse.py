import numpy as np
import matplotlib.pyplot as plt


# 读取数据，并根据类别分类
def readdata(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获取数据行数
    data = np.zeros((numberOfLines, 2))
    label = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split()
        data[index, 0] = float(listFromLine[0])
        data[index, 1] = float(listFromLine[1])
        label.append(float(listFromLine[-1]))
        index += 1
    # 分类
    index1 = np.array([index for (index, value) in enumerate(label) if value == -1.0])
    index2 = np.array([index for (index, value) in enumerate(label) if value == 1.0])
    data0 = data[index1]
    data1 = data[index2]
    return data0, data1


def calculatesi(datai, ui):
    si = np.zeros((datai.shape[1], datai.shape[1]))
    for xi in datai:
        m = xi - ui
        si += m * m.reshape(2, 1)
    return si


def fish(data0, data1):
    # 计算均值向量ui
    u0 = np.mean(data0, axis = 0)
    u1 = np.mean(data1, axis = 0)
    # 计算类内离散度矩阵si
    s0 = calculatesi(data0, u0)
    s1 = calculatesi(data1, u1)
    # 总类内离散度矩阵
    sw = s0 + s1
    # 求逆
    sw_inv = np.linalg.inv(sw)
    # 计算投影w
    w = np.dot(sw_inv, (u0 - u1))
    w0 = (np.dot(w.T, u0) + np.dot(w.T, u0)) / 2
    return w, u0, u1


def judge(filename, w, u0, u1):
    # 读取数据
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获取数据行数
    test_data = np.zeros((numberOfLines, 2))
    label = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split()
        test_data[index, 0] = float(listFromLine[0])
        test_data[index, 1] = float(listFromLine[1])
        index += 1
    # 判断类别
    center_0 = np.dot(w.T, u0)
    center_1 = np.dot(w.T, u1)
    for s in test_data:
        y = np.dot(w.T, s)
        # print(y0)
        if abs(y - center_0) < abs(y - center_1):
            label.append(-1.0)
        else:
            label.append(1.0)
    
    # 分类
    index1 = np.array([index for (index, value) in enumerate(label) if value == -1.0])
    index2 = np.array([index for (index, value) in enumerate(label) if value == 1.0])
    test_data0 = test_data[index1]
    test_data1 = test_data[index2]
    return test_data0, test_data1


def draw(data0, data1, w):
    plt.scatter(data0[:, 0], data0[:, 1], c = 'red', marker = 'x')
    plt.scatter(data1[:, 0], data1[:, 1], c = 'blue', marker = 'x')
    plt.show()


if __name__ == '__main__':
    # 读取数据集并根据数据类别分类
    data0, data1 = readdata("../data/fisher/train_data.txt")
    
    # 计算最佳投影w
    w, u0, u1 = fish(data0, data1)
    # 判断测试集
    test_data0, test_data1 = judge("../data/fisher/test_data.txt", w, u0, u1)
    # 绘图
    draw(test_data0, test_data1, w)
