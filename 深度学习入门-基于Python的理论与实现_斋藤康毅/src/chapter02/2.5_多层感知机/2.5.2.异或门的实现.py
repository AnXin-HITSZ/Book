import numpy as np

# 使用权重和偏置，可以像下面这样实现与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 使用权重和偏置，可以像下面这样实现与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置与 AND 不同！
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 使用权重和偏置，可以像下面这样实现或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])    # 仅权重和偏置与 AND 不同！
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 使用之前定义的 AND 函数、NAND 函数、OR 函数，可以像下面这样（轻松地）实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# 测试异或门
print(XOR(0, 0))    # 输出：0
print(XOR(1, 0))    # 输出：1
print(XOR(0, 1))    # 输出：1
print(XOR(1, 1))    # 输出：0
