# 비율 검정 : 집단의 비율이 어떤 특정한 값과 같은지를 검정(비율을 구해서 같은지 다른지 확인하는)
 
# one-sample
# A회사에서는 100명 중에 45명이 흡연을 한다. 국가 통게에서는 35%로 알려져 있다고 할 때, 두 그룹 간 비율이 동일한가?
# 귀무가설 : A회사의 흡연율과 국민 흡연율의 비율은 같다.
# 대립가설 : A회사의 흡연율과 국민 흡연율의 비율은 다르다.

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = np.array([45])
nobs = np.array([100])
val = 0.35
z,p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)  # [2.01007563]
print(p)  # [0.04442318] < 0.05 이므로 귀무가설 기각, A회사의 흡연율과 국민 흡연율의 비율은 다르다.

print()
# two-sample
# A회사 직원들 300명 중 100명이 햄버거를 먹었고, B회사 직원들 400명 중 170명이 햄버거를 먹었다고 할 때 비율의 동질 여부를 검정
count = np.array([100, 170])
nobs = np.array([300, 400])
z,p = proportions_ztest(count=count, nobs=nobs, value=0)
print(z) # -2.4656701201792273
print(p) # 0.013675721698622408  < 0.05 보다 작으므로 귀무가설 기각, 두 회사의 햄버거 취식 비율은 다르다.
