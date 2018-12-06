# -*-coding:utf-8-*-
"""
@Author: Rider
@Data: 2018-12-03
@File: naive-bayes.py
"""
import numpy as np
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


X, Y = get_iris()
len_data = len(X)
split_index = int(len_data*0.8)
# 划分数据集为训练集、验证集
X_train, X_test = X[:split_index, :], X[split_index:, :]
Y_train, Y_test = Y[:split_index], Y[split_index:]
labels = list(set(Y))
Py = []
# 计算所有P(Y)的先验概率
for l in labels:
    Py.append(np.sum(Y == labels[0])/np.sum(Y))
# print(labels)
# print(Py)


# 预测
def predict(X_train, Y_train, x):
    # m为特征的维度
    m = len(x)
    P = 0
    label = labels[0]
    for l in labels:
        PXPl = 1
        # 获取P(Y=l)的先验概率
        # 得到Y=l的所有下标
        indexl = np.where(Y_train == labels[l])[0]
        # 对每一个特征遍历
        for j in range(m):
            Ij = 0
            # 取出当前特征列第j列
            Xj = X_train[:, j]
            for i in indexl:
                # 在P(Y=l)的条件找到(X=x)的情况
                if Xj[i] == x[j]:
                    Ij += 1
            # 计算P(X=x)|P(Y=l)条件概率
            PxPl = Ij/len(indexl)
            # 计算累乘∏P(Xj=xj)|P(Y=l)
            PXPl *= PxPl
        # 计算P(Y=l)·∏P(Xj=xj)|P(Y=l)比较后验概率
        Pl = Py[l]*PXPl
        print('P(Y='+str(l)+'|X)', Pl)
        if Pl>P:
            P = Pl
            label = labels[l]
    return label


# 将所有特征元素四舍五入，相当于做了特征化
X_train_int = np.rint(X_train)
X_test_int = np.rint(X_test)


# 对预测准确率进行验证
def valid(X_train, X_test):
    I_true = 0
    for k in range(len(Y_test)):
        print('='*20)
        print(X_test[k])
        print(Y_test[k])
        prediction = predict(X_train, Y_train, X_test[k])
        print(prediction)
        if Y_test[k]==prediction:
            print('预测成功！！+1')
            I_true += 1
    print('='*20)
    print('*'*20)
    print('预测准确率为' + '%.2f' % (100*I_true/len(Y_test)) + '%')


if __name__ == '__main__':
    # 连续型数据
    valid(X_train, X_test)
    # 离散化数据
    valid(X_train_int, X_test_int)


"""
实验结论：朴素贝叶斯对于连续行特征的预测情况一般。
实验改进：将连续型数值特征进行label化，离散后情况是否好.
预测准确率： 连续型 80.00%   73.33%
            离散型 93.33%   96.67%
每次结果不同是因为数据集每次都打乱，排列顺序随机造成的。
如果数据集一定的情况下，预测结果应该不变。
朴素贝叶斯更加适用于离散数据！
"""