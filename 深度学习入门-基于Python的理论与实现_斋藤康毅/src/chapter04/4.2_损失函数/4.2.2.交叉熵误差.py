import numpy as np

# 实现交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7    # 作为保护性对策，添加一个微小值可以防止负无限大的发生
    return -np.sum(t * np.log(y + delta))

# 下面，我们使用 cross_entropy_error(y, t) 进行一些简单的计算
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))    # 输出：0.510825457099338

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))    # 输出：2.302584092994546
