import numpy as np

# 2 * 2 形状的矩阵的乘积
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

# 2 * 3 的矩阵和 3 * 2 的矩阵的乘积
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)
print(np.dot(A, B))

# 在多维数组的乘积运算中，必须使两个矩阵中的对应维度的元素个数一致
# # 以下无法计算矩阵的乘积
# C = np.array([[1, 2], [3, 4]])
# print(C.shape)
# print(A.shape)
# print(np.dot(A, C))

# 二维矩阵和一维数组的乘积
# 当 A 是二维矩阵、B 是一维数组时，对应维度的元素个数要保持一致的原则依然成立
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)
B = np.array([7, 8])
print(B.shape)
print(np.dot(A, B))
