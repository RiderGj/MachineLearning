# -*- coding:utf-8 -*-
"""
@Author: Rider
@Data: 2018-12-05
@File: decision-tree.py
"""
import numpy as np
import math
from sklearn import datasets


# 获取打乱的鸢尾花数据集
def get_iris():
    # 鸢尾花数据集
    iris = datasets.load_iris()
    X_iris = iris.get('data')
    Y_iris = iris.get('target')
    
    # 打乱数据集
    len_iris = len(Y_iris)
    arr_index = np.arange(len_iris)
    # 将下标打乱
    np.random.shuffle(arr_index)
    X_shf = np.array([X_iris[arr_index[i]] for i in arr_index])
    Y_shf = np.array([Y_iris[arr_index[i]] for i in arr_index])
    return X_shf, Y_shf


dataset = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 2, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 2, 1],
    [1, 0, 1, 2, 1],
    [2, 0, 1, 2, 1],
    [2, 0, 1, 1, 1],
    [2, 1, 0, 1, 1],
    [2, 1, 0, 2, 1],
    [2, 0, 0, 0, 0]
]

dataset = np.array(dataset)
dataY = dataset[:, 4]
labels = list(set(dataY))


# 计算数据集D的经验熵H(D)，都是对类别的经验熵
def cal_HD(data):
    dataY_temp = data[:, -1]
    HD = 0
    for i in labels:
        Pi = np.sum(dataY_temp == i) / len(dataY_temp)
        if Pi == 0:
            HD -= 0
        else:
            HD -= Pi * math.log2(Pi)
    return HD


# 计算HD_A
def cal_HD_A(k, data):
    feature_A = data[:, k]
    class_A = list(set(feature_A))
    HD_A = 0
    for i in range(len(class_A)):
        indexs_i = np.where(feature_A == class_A[i])[0]
        data_i = np.array([data[ix] for ix in indexs_i])
        # print('data_i', data_i)
        # print('indexs_i', indexs_i)
        # print(len(indexs_i)/len(feature_A))
        # print('HDi', cal_HD(data_i))
        HD_A += (len(indexs_i)/len(feature_A))*cal_HD(data_i)
    return HD_A



tree = {}
# 信息增益gain


def split_tree(data, feature_list=[0, 1, 2, 3, 4]):
    print(type(data[0]))
    if len(feature_list) == 0 or isinstance(data[0], np.ndarray) is False:
        if np.sum(data[:, -1] == 1)/len(data) > 0.5:
            return 1
        else:
            return 0
    hd = cal_HD(data)
    gain_max = hd-cal_HD_A(0, data)
    point_split = 0
    # 找到信息增益最大的特征，作为划分特征
    for j in range(len(data[0])-1):
        gain = hd-cal_HD_A(j, data)
        if gain > gain_max:
            gain_max = gain
            point_split = j
        print('特征' + str(j) + '的信息增益：' + str(gain))
    print('当前最大信息增益为：', gain_max)
    print('选择划分特征为特征', feature_list[point_split])
    # 在可划分特征中去除该特征
    del feature_list[point_split]
    feature_split = data[:, point_split]
    class_split = list(set(feature_split))
    tree_child = {}
    for cs in class_split:
        indexs_cs = np.where(feature_split == cs)[0]
        data_cs = np.array([data[ix] for ix in indexs_cs])
        tree_child[cs] = split_tree(data_cs, feature_list)
    return tree_child
    

if __name__ == '__main__':
    tree = split_tree(dataset, [0, 1, 2, 3, 4])
    print(tree)

"""
发现问题：
1、训练时：当决策树分到h层时，此时剩余p个数据点，信息增益最大的那个特征A共有3个分类0、1、2，
           但此时A=0的数据没有，剩余p个点都是A=1或2的。构造下一层子树的时候，只会有2个子节点1、2。
   测试时：当测试数据走到h层时，如果测试数据中A=0，就会出错，不知道分到哪一类。
   结论：所以ID3和C4.5在这种情况表现不好，而CART只由是或不是两种分类，构建二叉树则不会出现这种错误。
"""
# HD_A =
# print(datasets.load_breast_cancer())
# print(get_iris())
