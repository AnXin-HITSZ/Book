import numpy as np

# 针对溢出问题的具体示例
a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a)))    # softmax 函数的运算
# 上述代码运算结果为 array([nan, nan, nan])，由于溢出没有被正确计算

c = np.max(a)   # 1010
print(a - c)

print(np.exp(a - c) / np.sum(np.exp(a - c)))

# 综上，我们可以像下面这样实现 softmax 函数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)   # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
