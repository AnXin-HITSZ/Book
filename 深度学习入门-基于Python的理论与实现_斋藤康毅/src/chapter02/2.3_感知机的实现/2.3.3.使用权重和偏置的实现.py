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
