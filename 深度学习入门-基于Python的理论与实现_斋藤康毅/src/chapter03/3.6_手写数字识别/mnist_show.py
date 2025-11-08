import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    # 此外，还需要把保存为 NumPy 数组的图像数据转换为 PIL 用的数据对象
    # 这个转换处理由 Image.fromarray() 来完成
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 第一次调用会花费几分钟……
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 输出各个数据的形状
print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000,)
print(x_test.shape)     # (10000, 784)
print(t_test.shape)     # (10000,)

img = x_train[0]
label = t_train[0]
print(label)    # 5

# flatten=True 时读入的图像是以一列（一维）NumPy 数组的形式保存的
print(img.shape)    # (784,)
# 因此，显示图像时，需要把它变为原来的 28 像素 × 28 像素的形状
img = img.reshape(28, 28)   # 把图像的形状变成原来的尺寸
print(img.shape)    # (28, 28)

img_show(img)
