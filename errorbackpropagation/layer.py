# 激活函数层的实现
import numpy as np

from common.functions import softmax, cross_entropy_error


# Relu层
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 计算relu激活后的值
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # 计算relu激活后的导数
        dout[self.mask] = 0
        dx = dout

        return dx


# Sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # sigmoid函数：计算Sigmoid激活值
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        # 计算Sigmoid函数的导数
        dx = dout * (1 - self.out) * self.out
        return dx


# 全连接层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# SoftmaxWithLoss层
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    # 损失
        self.y = None   # softmax的输出
        self.t = None   # 监督数据(one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

