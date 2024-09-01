# 损失函数
import numpy as np


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 设“2”为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 例1：“2"的概率最高
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y1), np.array(t)))    # 0.09750000000000003
# 例2：“7”的概率最高
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y2), np.array(t)))    # 0.5975


# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# # 设“2”为正确解
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# # 例1：“2"的概率最高
# y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y1), np.array(t)))    # 0.510825457099338
# # 例2：“7”的概率最高
# y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y2), np.array(t)))    # 2.302584092994546

