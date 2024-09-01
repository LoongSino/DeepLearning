# mini_batch学习
import sys, os
import numpy as np
from dataset.mnist import load_mnist

sys.path.append(os.pardir)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # 随机选择10个数字
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# mini-batch版交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]
    # one-hot表示
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # 标签形式(非one-hot表示)
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
