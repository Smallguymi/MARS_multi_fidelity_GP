import math
import torch
import gpytorch
import matplotlib.pyplot as plt
import os
import numpy as np


#%%


'''
2025.1.2 test

Train_x1=[500,240,1.44,30]	#	[NPSI,NCHI,CFBAL,M2]
Train_y1=[1.89476E-03]	#	[GR]

Train_x2=[500,240,1.46,30]	#	[NPSI,NCHI,CFBAL,M2]
Train_y2=[4.36939E-03]	#	[GR]


'''

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 定义数据
Train_x1 = [[500, 240, 1.44, 30]]  # 注意：输入数据需要是二维列表
Train_y1 = [1.89476E-03]
Train_x2 = [[500, 240, 1.46, 30]]
Train_y2 = [4.36939E-03]

# 合并数据
X_train = Train_x1 + Train_x2
y_train = Train_y1 + Train_y2

# 创建高斯过程回归模型
# 这里使用径向基函数（RBF）核，你也可以尝试其他核函数
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# 训练模型
gpr.fit(X_train, y_train)

# 定义测试数据
X_test = [[500, 240, 1.45, 30]]  # 这里是一个示例测试点

# 进行预测
y_pred, sigma = gpr.predict(X_test, return_std=True)

print(f"预测结果: {y_pred}")
print(f"预测的标准差: {sigma}")



#%%

#以下是读取训练所需的数据的函数



import os
import re

# 设置目录路径
directory_path = r'F:\MARS_surrogate_model'

# 初始化一个空列表来存储提取的值
values = []

# 编译正则表达式以匹配文件夹名称
pattern = re.compile(r'NPSI_(\d+)_NCHI_(\d+)')

# 遍历目录下的所有文件夹
for folder_name in os.listdir(directory_path):
    # 检查是否为文件夹
    if os.path.isdir(os.path.join(directory_path, folder_name)):
        # 使用正则表达式匹配文件夹名称
        match = pattern.match(folder_name)
        if match:
            # 如果匹配成功，提取*的值
            npsi_value, nchi_value = match.groups()
            # 将提取的值添加到列表中
            values.append((npsi_value, nchi_value))

# 打印结果
print(values)


def inputdata(s_path):
    # 构造文件路径
    in_path = f"{s_path}/bin/extkink.txt"
    
    # 打开文件
    with open(in_path, 'r') as fid:
        # 读取第一行数据，假设数据是以空格分隔的
        data1 = fid.readline().strip().split()
    
    # 初始化NN列表
    NN = []
    
    # 查找特定关键词的索引
    keywords = ['CFBAL', 'CSSPEC', 'QSPEC', 'ETA', 'TALPHA1_REAL', 'TALPHA2_IMAG']
    for keyword in keywords:
        try:
            # 找到关键词的索引并添加到NN列表中
            index = data1.index(keyword)
            NN.append(index)
        except ValueError:
            # 如果关键词不在列表中，添加一个特殊值，比如-1
            NN.append(-1)
    
    return data1, NN

# 使用示例
# s_path = 'your_path_here'  # 替换为你的路径
# data1, NN = inputdata(s_path)
# print("Data:", data1)
# print("Indices:", NN)







































