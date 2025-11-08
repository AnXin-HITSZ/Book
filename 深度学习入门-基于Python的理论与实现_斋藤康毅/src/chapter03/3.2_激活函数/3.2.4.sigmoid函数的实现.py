import numpy as np
import matplotlib.pylab as plt

# sigmoid 函数的实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 绘制 sigmoid 函数的图像
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定 y 轴的范围
plt.show()
