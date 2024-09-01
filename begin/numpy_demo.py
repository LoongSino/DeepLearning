import numpy as np
from numpy import ndarray


x = np.array([1.0, 2.0, 3.0])

print(x)    # [1. 2. 3.]
print(type(x))  # <class 'numpy.ndarray'>

y = np.array([2.0, 4.0, 6.0])

# NumPy的算术运算
print(x + y)    # [3. 6. 9.]
print(x - y)    # [-1. -2. -3.]
print(x * y)    # [ 2.  8. 18.]
print(x / y)    # [0.5 0.5 0.5]

print(x/2.0)    # [0.5 1.  1.5]


# NumPy的N维数组
A = np.array([[1, 2], [3, 4]])
print(A)
# [[1 2]
#  [3 4]]
print(A.shape)  # 矩阵的形状 (2, 2)
print(A.dtype)  # 矩阵元素的数据类型 int64

B = np.array([[3, 0], [0, 6]])

print(A + B)
# [[ 4  2]
#  [ 3 10]]
print(A * B)
# [[ 3  0]
#  [ 0 24]]

print(A * 10)
# [[10 20]
#  [30 40]]

# 广播：形状不同的数组之间的运算
C = np.array([10, 20])

print(A * C)
# [[10 40]
#  [30 80]]

# 访问元素
X = np.array([[51, 55], [14, 19], [0, 4]])

print(X)
# [[51 55]
#  [14 19]
#  [ 0  4]]
print(X[0])     # 第0行 [51 55]
print(X[:, 0])  # 第0列 [51 14  0]
print(X[0][1])  # 55


for row in X:
    print(row)
# [51 55]
# [14 19]
# [0 4]

X = X.flatten()     # 将X转换为一维数组
print(X)    # [51 55 14 19  0  4]
print(X[np.array([0, 2, 4])])   # [51 14  0]
print(X > 15)   # [ True  True False  True False False]
print(X[X > 15])    # [51 55 19]

