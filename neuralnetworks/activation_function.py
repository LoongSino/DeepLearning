# 激活函数

import numpy as np
import matplotlib.pylab as plt


# 阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int64)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU函数
def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)

y1 = step_function(x)
plt.plot(x, y1, label="step", linestyle="--")
y2 = sigmoid(x)
plt.plot(x, y2, label="sigmoid")
y3 = relu(x)
plt.plot(x, y3, label="relu")

plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.legend(loc="upper left")
plt.show()

