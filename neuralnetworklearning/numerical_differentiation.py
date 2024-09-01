# 数值微分
import numpy as np
import matplotlib.pylab as plt


# 中心差分
def numerical_diff(f, x):
    h = 1e-4    # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))    # 0.1999999999990898
print(numerical_diff(function_1, 10))   # 0.2999999999986347


def function_2(x):
    return np.sum(x**2)


# 梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)     # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val    # 还原值

    return grad


print(numerical_gradient(function_2, np.array([3.0, 4.0])))     # [6. 8.]
print(numerical_gradient(function_2, np.array([0.0, 4.0])))     # [0. 8.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))     # [6. 0.]


# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))    # [-6.11110793e-10  8.14814391e-10]
