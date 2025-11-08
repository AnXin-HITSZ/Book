import numpy as np
import matplotlib.pylab as plt

# 阶跃函数的实现
def step_function(x):
    return np.array(x > 0, dtype=np.int_)

# 绘制阶跃函数的图像
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定 y 轴的范围
plt.show()
