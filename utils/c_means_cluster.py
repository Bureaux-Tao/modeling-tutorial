#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:51:45 2019
模糊c聚类:https://blog.csdn.net/lyxleft/article/details/88964494
@author: youxinlin
"""
import copy
import math
import random
import time

import numpy
import numpy as np
from sklearn.metrics import calinski_harabasz_score

global MAX  # 用于初始化隶属度矩阵U
MAX = 10000.0

global Epsilon  # 结束条件
Epsilon = 0.0000001


def print_matrix(list):
    """
    以可重复的方式打印矩阵
    """
    for i in range(0, len(list)):
        print(list[i])


def initialize_U(data, cluster_number):
    """
    这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    """
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):
    """
    该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    """
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    """
    在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
    """
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def c_means_fuzzy(data, cluster_number, m, hard_max=True):
    """
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
    """
    # 初始化隶属度矩阵U
    U = initialize_U(data, cluster_number)
    # print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("已完成聚类")
            break

    if hard_max:
        U = normalise_U(U)
    return U


def de_randomise_data(data, order):
    """
    此函数将返回数据的原始顺序，将randomise_data()返回的order列表作为参数
    """
    new_data = [[] for i in range(0, len(data))]
    for index in range(len(order)):
        new_data[order[index]] = data[index]
    return new_data


def get_accuracy(arr1, arr2):
    # arr1, arr2都是np.array类型
    # 找到重复元素（交集）
    if arr1.shape != arr2.shape:
        raise Exception("不同型矩阵!")
    inters = np.intersect1d(arr1, arr2)
    # 元素个数索引转换
    bc1 = np.bincount(arr1)
    bc2 = np.bincount(arr2)
    # 统计相同元素匹配个数
    same_count_list = [min(bc1[x], bc2[x]) for x in inters]
    same_count = sum(same_count_list)
    return same_count / len(arr1)


if __name__ == '__main__':
    data = [[6.1, 2.8, 4.7, 1.2], [5.1, 3.4, 1.5, 0.2], [6.0, 3.4, 4.5, 1.6], [4.6, 3.1, 1.5, 0.2],
            [6.7, 3.3, 5.7, 2.1], [7.2, 3.0, 5.8, 1.6], [6.7, 3.1, 4.4, 1.4], [6.4, 2.7, 5.3, 1.9],
            [4.8, 3.0, 1.4, 0.3], [7.9, 3.8, 6.4, 2.0], [5.2, 3.5, 1.5, 0.2], [5.9, 3.0, 5.1, 1.8],
            [5.7, 2.8, 4.1, 1.3], [6.8, 3.2, 5.9, 2.3], [5.4, 3.4, 1.5, 0.4], [5.4, 3.7, 1.5, 0.2],
            [6.6, 3.0, 4.4, 1.4], [5.1, 3.5, 1.4, 0.2], [6.0, 2.2, 4.0, 1.0], [7.7, 2.8, 6.7, 2.0],
            [6.3, 2.8, 5.1, 1.5], [7.4, 2.8, 6.1, 1.9], [5.5, 4.2, 1.4, 0.2], [5.7, 3.0, 4.2, 1.2],
            [5.5, 2.6, 4.4, 1.2], [5.2, 3.4, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [4.6, 3.6, 1.0, 0.2],
            [4.6, 3.2, 1.4, 0.2], [5.8, 2.7, 3.9, 1.2], [5.0, 3.4, 1.5, 0.2], [6.1, 3.0, 4.6, 1.4],
            [4.7, 3.2, 1.6, 0.2], [6.7, 3.3, 5.7, 2.5], [6.5, 3.0, 5.8, 2.2], [5.4, 3.4, 1.7, 0.2],
            [5.8, 2.7, 5.1, 1.9], [5.4, 3.9, 1.3, 0.4], [5.3, 3.7, 1.5, 0.2], [6.1, 3.0, 4.9, 1.8],
            [7.2, 3.2, 6.0, 1.8], [5.5, 2.3, 4.0, 1.3], [5.7, 2.8, 4.5, 1.3], [4.9, 2.4, 3.3, 1.0],
            [5.4, 3.0, 4.5, 1.5], [5.0, 3.5, 1.6, 0.6], [5.2, 4.1, 1.5, 0.1], [5.8, 4.0, 1.2, 0.2],
            [5.4, 3.9, 1.7, 0.4], [6.5, 3.2, 5.1, 2.0], [5.5, 2.4, 3.7, 1.0], [5.0, 3.5, 1.3, 0.3],
            [6.3, 2.5, 5.0, 1.9], [6.9, 3.1, 4.9, 1.5], [6.2, 2.2, 4.5, 1.5], [6.3, 3.3, 4.7, 1.6],
            [6.4, 3.2, 4.5, 1.5], [4.7, 3.2, 1.3, 0.2], [5.5, 2.4, 3.8, 1.1], [5.0, 2.0, 3.5, 1.0],
            [4.4, 2.9, 1.4, 0.2], [4.8, 3.4, 1.9, 0.2], [6.3, 3.4, 5.6, 2.4], [5.5, 2.5, 4.0, 1.3],
            [5.7, 2.5, 5.0, 2.0], [6.5, 3.0, 5.2, 2.0], [6.7, 3.0, 5.0, 1.7], [5.2, 2.7, 3.9, 1.4],
            [6.9, 3.1, 5.1, 2.3], [7.2, 3.6, 6.1, 2.5], [4.8, 3.0, 1.4, 0.1], [6.3, 2.9, 5.6, 1.8],
            [5.1, 3.5, 1.4, 0.3], [6.9, 3.1, 5.4, 2.1], [5.6, 3.0, 4.1, 1.3], [7.7, 2.6, 6.9, 2.3],
            [6.4, 2.9, 4.3, 1.3], [5.8, 2.7, 4.1, 1.0], [6.1, 2.9, 4.7, 1.4], [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.8, 4.8, 1.8], [4.8, 3.4, 1.6, 0.2], [5.6, 2.9, 3.6, 1.3], [6.7, 2.5, 5.8, 1.8],
            [5.0, 3.4, 1.6, 0.4], [6.3, 3.3, 6.0, 2.5], [5.1, 3.8, 1.9, 0.4], [6.6, 2.9, 4.6, 1.3],
            [5.1, 3.3, 1.7, 0.5], [6.3, 2.5, 4.9, 1.5], [6.4, 3.1, 5.5, 1.8], [6.2, 3.4, 5.4, 2.3],
            [6.7, 3.1, 5.6, 2.4], [4.6, 3.4, 1.4, 0.3], [5.5, 3.5, 1.3, 0.2], [5.6, 2.7, 4.2, 1.3],
            [5.6, 2.8, 4.9, 2.0], [6.2, 2.9, 4.3, 1.3], [7.0, 3.2, 4.7, 1.4], [5.0, 3.2, 1.2, 0.2],
            [4.3, 3.0, 1.1, 0.1], [7.7, 3.8, 6.7, 2.2], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 5.1, 1.9],
            [5.8, 2.8, 5.1, 2.4], [4.9, 3.1, 1.5, 0.1], [5.7, 3.8, 1.7, 0.3], [7.1, 3.0, 5.9, 2.1],
            [5.1, 3.7, 1.5, 0.4], [6.3, 2.7, 4.9, 1.8], [6.7, 3.0, 5.2, 2.3], [5.1, 2.5, 3.0, 1.1],
            [7.6, 3.0, 6.6, 2.1], [4.5, 2.3, 1.3, 0.3], [4.9, 3.0, 1.4, 0.2], [6.5, 2.8, 4.6, 1.5],
            [5.7, 4.4, 1.5, 0.4], [6.8, 3.0, 5.5, 2.1], [4.9, 2.5, 4.5, 1.7], [5.1, 3.8, 1.5, 0.3],
            [6.5, 3.0, 5.5, 1.8], [5.7, 2.6, 3.5, 1.0], [5.1, 3.8, 1.6, 0.2], [5.9, 3.0, 4.2, 1.5],
            [6.4, 3.2, 5.3, 2.3], [4.4, 3.0, 1.3, 0.2], [6.1, 2.8, 4.0, 1.3], [6.3, 2.3, 4.4, 1.3],
            [5.0, 2.3, 3.3, 1.0], [5.0, 3.6, 1.4, 0.2], [5.9, 3.2, 4.8, 1.8], [6.4, 2.8, 5.6, 2.2],
            [6.1, 2.6, 5.6, 1.4], [5.6, 2.5, 3.9, 1.1], [6.0, 2.7, 5.1, 1.6], [6.0, 3.0, 4.8, 1.8],
            [6.4, 2.8, 5.6, 2.1], [6.0, 2.9, 4.5, 1.5], [5.8, 2.6, 4.0, 1.2], [7.7, 3.0, 6.1, 2.3],
            [5.0, 3.3, 1.4, 0.2], [6.9, 3.2, 5.7, 2.3], [6.8, 2.8, 4.8, 1.4], [4.8, 3.1, 1.6, 0.2],
            [6.7, 3.1, 4.7, 1.5], [4.9, 3.1, 1.5, 0.1], [7.3, 2.9, 6.3, 1.8], [4.4, 3.2, 1.3, 0.2],
            [6.0, 2.2, 5.0, 1.5], [5.0, 3.0, 1.6, 0.2]]
    start = time.time()

    # 调用模糊C均值函数
    res_U = c_means_fuzzy(data, 3, 2)
    # print_matrix(res_U)
    # 计算准确率
    print("用时：{0}".format(time.time() - start))
    print("-------------------------------------")
    wine = np.loadtxt("../data/wine.data", dtype=float, delimiter=',')
    y = wine[:, 0].astype(int)
    x = wine[:, 1:]
    print(x.shape)
    print(y.shape)

    wine_U = c_means_fuzzy(wine, 3, 2)
    y_prd = np.array(wine_U).argmax(axis=1)
    print(get_accuracy(y, y_prd))

    print("-------------------------------------")

    iris = []
    cluster_location = []
    with open(str("../data/iris.txt"), 'r') as f:
        tmp = []
        for line in f:
            tmp.append(line.strip())
        random.shuffle(tmp)
        # print(tmp)

    for line in tmp:
        current = line.strip().split(",")  # 对每一行以逗号为分割，返回一个list
        current_dummy = []
        for j in range(0, len(current) - 1):
            current_dummy.append(float(current[j]))  # current_dummy存放data

        j += 1
        if current[j] == "Iris-setosa":
            cluster_location.append(0)
        elif current[j] == "Iris-versicolor":
            cluster_location.append(1)
        else:
            cluster_location.append(2)
        iris.append(current_dummy)

    # print_matrix(iris)
    # print(cluster_location)
    iris_U = c_means_fuzzy(iris, 3, 2)
    iris_prd = np.array(iris_U).argmax(axis=1)
    print(get_accuracy(np.array(cluster_location), numpy.array(iris_prd)))
    print(calinski_harabasz_score(iris, iris_prd))
