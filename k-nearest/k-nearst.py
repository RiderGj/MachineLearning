# -*- coding:utf-8 -*-
"""
@Author: Rider
@Data: 2018-11-27
@File: k-nearst.py
"""
import numpy as np
from sklearn import datasets

# 鸢尾花数据集
iris = datasets.load_iris()
X_iris = iris.get('data')
Y_iris = iris.get('target')
# print(iris)
print(np.array(X_iris))
print(np.array(Y_iris))
X = np.array([
    [1, 1],
    [5, 1],
    [4, 4],
    [-1, 3],
    [1, -6],
    [3, -1]
])
Y = np.array([1, 2, 1, 3, 1, 2])
l = len(X_iris)
X_train, X_test = X_iris[:int(l*0.8), :], X_iris[int(l*0.8):, :]
Y_train, Y_test = Y_iris[:int(l*0.8)], Y_iris[int(l*0.8):]
k = 10


# Lp距离
def dist(x1, x2, p):
    return np.power(np.sum(np.power(abs(x1-x2), p)), 1/p)


def max_list(lt):
    temp = 0
    max_str = lt[0]
    for i in set(lt):
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def predict(x, k):
    dists = []
    for i in range(len(X_train)):
        dists.append(dist(X_train[i], x, 2))
    indexs_sorted = np.argsort(dists)
    targets = [Y_train[i] for i in indexs_sorted[:k]]
    return max_list(targets)


x1= [6.5, 3.,  5.2, 2. ]
print(predict(x1, 10))
