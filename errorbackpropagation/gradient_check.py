# 误差反向传播法的梯度确认
import os
import sys
import numpy as np
from dataset.mnist import load_mnist
from neuralnetworklearning.two_layer_net import TwoLayerNet

sys.path.append(os.pardir)

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

    # W1:1.791163221067509e-10
    # b1:8.536566677949967e-10
    # W2:7.126199243326359e-08
    # b2:1.4253074291231683e-07

