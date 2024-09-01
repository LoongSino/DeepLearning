# 多维数组的运算
import numpy as np


# 一维数组
A = np.array([1, 2, 3, 4])
print(A)    # [1 2 3 4]
print(np.ndim(A))   # 矩阵的维数 1
print(A.shape)  # (4,)
print(A.shape[0])   # 4

# 二维数组
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
# [[1 2]
#  [3 4]
#  [5 6]]
print(np.ndim(B))   # 矩阵的维数 2
print(B.shape)  # (3, 2) 三行四列

# 矩阵的点积
A1 = np.array([[1, 2], [3, 4]])
B1 = np.array([[5, 6], [7, 8]])
print(np.dot(A1, B1))
# [[19 22]
#  [43 50]]

A2 = np.array([[1, 2, 3], [4, 5, 6]])
B2 = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(A2, B2))
# [[22 28]
#  [49 64]]

A3 = np.array([[1, 2], [3, 4], [5, 6]])
B3 = np.array([7, 8])
print(np.dot(A3, B3))   # [23 53 83]

# 神经网络的内积
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
print(np.dot(X, W))     # [ 5 11 17]

