"""
介绍利用 NumPy 来生成多维数组的实现
"""

import numpy as np

# 一维数组的情况
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))   # 获取数组的维数
print(A.shape)  # 获取数组的形状，返回的结果为元组（tuple）
print(A.shape[0])

# 二维数组的情况
# 二维数组也称为矩阵（matrix）
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
