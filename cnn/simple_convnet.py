import sys, os
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size': 5, 'stride':1, 'pad':0},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_stride = conv_param['stride']
        filter_pad = conv_param['pad']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {'W1': weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size),
                       'b1': np.zeros(filter_num),
                       'W2': weight_init_std * np.random.randn(pool_output_size, hidden_size),
                       'b2': np.zeros(hidden_size),
                       'W3': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b3': np.zeros(output_size)}

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {'W1': self.layers['Conv1'].dW,
                 'b1': self.layers['Conv1'].db,
                 'W2': self.layers['Affine1'].dW,
                 'b2': self.layers['Affine1'].db,
                 'W3': self.layers['Affine2'].dW,
                 'b3': self.layers['Affine2'].db}

        return grads

