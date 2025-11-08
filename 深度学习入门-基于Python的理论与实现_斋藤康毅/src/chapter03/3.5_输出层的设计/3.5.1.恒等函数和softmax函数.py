import numpy as np

# 实现 softmax 函数
a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)   # 指数函数
print(exp_a)

sum_exp_a = np.sum(exp_a)   # 指数函数的和
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

# 考虑到后面还要使用 softmax 函数，将上述实现逻辑封装并定义成如下的 Python 函数
def softmax(a):
    exp_a = np.exp(a)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
