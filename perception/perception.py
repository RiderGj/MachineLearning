# -*- coding:utf-8 -*-
"""
@File: perception.py
@Author: Rider
@Date: 2018-11-26
"""
import numpy as np


# 感知器模型
def model_perception(x, y, w0, b0, n_iteration=20, learning_rate=0.1):
    w = w0
    b = b0
    for j in range(n_iteration):
        print('='*30)
        print('第' + str(j+1) + '次迭代。。。')
        for i in range(len(x)):
            loss = (np.dot(x[i], w.T) + b)*y[i]
            # 使用l来判断是否分类成功，只有l>0分类成功
            if loss <= 0:
                print('x'+str(i+1)+'未被正确分类')
                w = w + learning_rate*np.dot(x[i], y[i])
                b = b + learning_rate*y[i]
                print('w=', w)
                print('b=', b)
                break
            if i == len(x)-1:
                print('超平面构造完毕')
                func = 'y = '
                for k in range(len(w)):
                    func += '%.3f' % (w[k])+'x'+str(k+1)+' + '
                print('超平面函数为：' + func + str(b))
                return w, b


def predict_perception(w, b, x):
    pred = []
    for xi in x:
        res = np.dot(xi, w) + b
        if res >= 0:
            pred.append(1)
        else:
            pred.append(-1)
    return np.array(pred)


if __name__ == '__main__':
    x = np.array([[1, 3, 5], [5, 2, -6], [3, -1, -8], [3, 6, 9]])
    y = np.array([1, 1, -1, 1])
    # w0 = np.zeros(x.shape[1])
    w0 = np.array([1, -1, 1])
    b0 = 0
    w, b = model_perception(x, y, w0, b0, n_iteration=200, learning_rate=0.01)
    perdiction = predict_perception(w, b, x)
    print(perdiction)
    accuracy = np.array([1 if y[i] == perdiction[i] else 0 for i in range(len(x))]).sum()/len(x)
    print('分类准确率为： '+str(accuracy*100)+'%')
