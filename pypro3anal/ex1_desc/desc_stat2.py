# 표준편차, 분산의 중요성 : 평균은 같으나 분산이 다름으로 인해 전체 데이터의 분포상태가 달라진다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

np.random.seed(1)
print(stats.norm(loc = 1, scale=2).rvs(10)) # 정규분포를 따르는 랜덤표본 생성

print('-------')
centers = [1, 1.5, 2]
col = 'rgb'

std = 10 # 표준편차가 작아질수록 평균에 모임,, 커질수록 평균으로부터 퍼짐
datas = []

for i in range(3):
    datas.append(stats.norm(loc = centers[i], scale=std).rvs(100)) # 기댓값 centers[i], 표준편차 1, 랜덤변수100
    plt.plot(np.arange(100) + i * 100, datas[i], '*', color=col[i])

plt.show()