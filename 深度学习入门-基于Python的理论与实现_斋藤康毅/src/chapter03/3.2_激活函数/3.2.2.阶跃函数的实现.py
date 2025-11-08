import numpy as np

# # 可以像下面这样简单地实现阶跃函数
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 为了实现参数取 NumPy 数组的操作，将以上阶跃函数修改为如下支持 NumPy 数组的实现
def step_function(x):
    y = x > 0
    return y.astype(np.int)
