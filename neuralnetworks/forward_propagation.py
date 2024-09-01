# 各层间的信号传递 从输入层到输出层的传递过程
import numpy as np


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


# 输入层到第一层的信号传递 A = X * W +B
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
print(A1)   # [0.3 0.7 1.1]
Z1 = sigmoid(A1)
print(Z1)   # [0.57444252 0.66818777 0.75026011]


# 第一层到第二层的信号传递
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
print(A2)   # [0.51615984 1.21402696]
Z2 = sigmoid(A2)
print(Z2)   # [0.62624937 0.7710107 ]


# 第二层到输出层的信号传递
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
print(A3)   # [0.31682708 0.69627909]
Y = identity_function(A3)
print(Y)    # [0.31682708 0.69627909]

