import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess

# 테스트 데이터
trainX = np.array([1, 2, 3, 4, 5])
trainY = np.array([1, 2, 1.5, 2.5, 4])
h = 1.0

# x 값의 테스트 범위
x_values = np.linspace(0, 6, 100)

# Kernel Smoothing with 'statsmodels' KernelReg
# Gaussian 커널을 사용하여 커널 스무딩을 수행합니다.
kernel_reg = KernelReg(trainY, trainX, var_type='c', bw=[h])
smoothed_values, _ = kernel_reg.fit(x_values)

# Local Linear Regression using LOWESS
# 로컬 선형 회귀 (LOWESS)를 사용하여 결과를 예측합니다.
lowess_results = lowess(trainY, trainX, frac=0.3)  # frac은 bandwidth 역할

# 커널 스무딩 결과 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_values, smoothed_values, label='Kernel Smoothing (Gaussian)')
plt.scatter(trainX, trainY, color='red', label='Data Points')
plt.title('Kernel Smoothing with KernelReg (Gaussian)')
plt.xlabel('x')
plt.ylabel('Smoothed y')
plt.legend()

# 로컬 선형 회귀 결과 시각화
plt.subplot(1, 2, 2)
plt.plot(lowess_results[:, 0], lowess_results[:, 1], label='LOWESS')
plt.scatter(trainX, trainY, color='red', label='Data Points')
plt.title('Local Linear Regression with LOWESS')
plt.xlabel('x')
plt.ylabel('Estimated y')
plt.legend()

plt.tight_layout()
plt.show()
