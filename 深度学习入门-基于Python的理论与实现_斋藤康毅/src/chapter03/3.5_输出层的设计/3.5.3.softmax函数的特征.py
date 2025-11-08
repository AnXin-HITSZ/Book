import numpy as np

# softmax 函数的实现
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 溢出对策
    sum_exp_a = sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 使用 softmax() 函数，可以按如下方式计算神经网络的输出
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
