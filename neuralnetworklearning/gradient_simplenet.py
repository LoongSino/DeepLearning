# 简单的神经网络
import os
import sys
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

sys.path.append(os.pardir)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])     # 正确解标签

net = simpleNet()

f = lambda W: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)

