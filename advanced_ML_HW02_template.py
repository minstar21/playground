# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:26:40 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import linalg

#TODO: implement consine kernel
def cosine_kernel(x, xi, h):
    # x: test point
    # xi: i-th training point
    # h: bandwidth
    # return K((x-xi)/h)
    u = (x - xi) / h
    if abs(u) <= 1:
        return (np.pi / 4) * np.cos((np.pi / 2) * u)
    else:
        return 0
    
#TODO: implement gaussian kernel
def gaussian_kernel(x, xi, h):
    # x: test point
    # xi: i-th training point
    # h: bandwidth
    # return K((x-xi)/h)
    u = (x - xi) / h
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)    
   
#TODO: implement epanechnikov kernel
def epanechnikov_kernel(x,xi,h):
    # x: test point
    # xi: i-th training point
    # h: bandwidth
    # return K((x-xi)/h)
    u = (x - xi) / h
    if abs(u) <= 1:
        return 0.75 * (1 - u ** 2)
    else:
        return 0 
    
#TODO: implement triangular kernel
def triangular_kernel(x,xi,h):
    # x: test point
    # xi: i-th training point
    # h: bandwidth
    # return K((x-xi)/h)
    u = (x - xi) / h
    if abs(u) <= 1:
        return 1 - abs(u)
    else:
        return 0

#TODO: implement kernel smoothing function
def kernel_smoothing(x,trainX,trainY,kernel,h):
    # x: test point (scalar)
    # trainX: train input data (1D)
    # trainY: train output target
    # kernel: kernel function (use one of four kernel functions)
    # h: bandwidth
    # return predicted output for x
    numerator = 0
    denominator = 0
    for i in range(len(trainX)):
        K = kernel(x, trainX[i], h)
        numerator += K * trainY[i]
        denominator += K
    return numerator / denominator if denominator != 0 else 0
    
#TODO: implement local linear regression
def local_linear(x,trainX,trainY,kernel,h):
    # x: test point (scalar)
    # trainX: train input data (1D)
    # trainY: train output target
    # kernel: kernel function (use one of four kernel functions)
    # h: bandwidth
    # return predicted output for x
    n = len(trainX)
    X = np.vstack([np.ones(n), trainX - x]).T
    W = np.diag([kernel(x, xi, h) for xi in trainX])
    
    try:
        # β = (X.T W X)^-1 X.T W Y 계산
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ trainY
        return beta[0]
    except np.linalg.LinAlgError:
        # 행렬이 역행렬이 불가능한 경우 평균 반환
        return np.mean(trainY)
    
# Data generation
np.random.seed(1234)
trainX = np.random.uniform(0,1.5,200)
trainY_true = np.sin(trainX*np.pi)
trainY = trainY_true + 0.2*np.random.randn(*trainX.shape)


xx = np.linspace(0, 1.5, 200)
kernel_functions = [cosine_kernel, gaussian_kernel, epanechnikov_kernel,triangular_kernel]
kernel_names = ['Cosine', 'Gaussian', 'Epanechnikov', 'Triangular']

#TODO: Draw figure for kernel smoothing
h = 0.1  # 밴드위스 설정
# 커널 스무딩 결과 시각화
plt.figure(figsize=(15, 6))
for i, kernel in enumerate(kernel_functions):
    smoothed_values = [kernel_smoothing(x, trainX, trainY, kernel, h) for x in xx]
    plt.subplot(1, 4, i + 1)
    plt.plot(xx, smoothed_values, label=f'{kernel_names[i]} Kernel Smoothing')
    plt.scatter(trainX, trainY, color='red', label='Data Points')
    plt.title(f'{kernel_names[i]} Kernel Smoothing')
    plt.xlabel('x')
    plt.ylabel('Smoothed y')
    plt.legend()
plt.tight_layout()
plt.show()

    
#TODO: Draw figure for local linear regression
# 로컬 선형 회귀 결과 시각화
plt.figure(figsize=(15, 6))
for i, kernel in enumerate(kernel_functions):
    linear_reg_values = [local_linear(x, trainX, trainY, kernel, h) for x in xx]
    plt.subplot(1, 4, i + 1)
    plt.plot(xx, linear_reg_values, label=f'{kernel_names[i]} Local Linear Regression')
    plt.scatter(trainX, trainY, color='red', label='Data Points')
    plt.title(f'{kernel_names[i]} Local Linear Regression')
    plt.xlabel('x')
    plt.ylabel('Estimated y')
    plt.legend()
plt.tight_layout()
plt.show()