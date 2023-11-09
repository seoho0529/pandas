# 공분산 / 상관계수
import numpy as np

print(np.cov(np.arange(1,6), np.arange(2,7)))    # 2.5
print(np.cov(np.arange(1,6), (3,3,3,3,3)))       # 0
print(np.cov(np.arange(1,6), np.arange(6,1,-1))) # -2.5

print()
x = [8,3,6,6,9,4,3,9,4,3]
print('평균:', np.mean(x), ', 분산:',np.var(x))
# y = [6,2,4,6,9,5,1,8,4,5]
y = [600,200,400,600,900,500,100,800,400,500]
print('평균:', np.mean(y), ', 분산:',np.var(y))

import matplotlib.pyplot as plt
# plt.plot(x,y,'o')
plt.scatter(x,y)
plt.show()
# print('x,y의 공분산 : ',np.cov(x,y))
print('x,y의 공분산 : ',np.cov(x,y)[0,1]) # 공분산 0행 1열값만 보기

# print('x,y의 상관계수 : ', np.corrcoef(x,y))
print('x,y의 상관계수 : ',np.corrcoef(x,y)[0,1]) # 0.8479 이므로 매우 강한 양의 상관관계를 띄고 있다.(밀도가 높다, 데이터들이 추세선을 기준으로 촘촘히 모여있다.)
# 상관계수는 두 변수 간의 선형 관계를 파악하는 데 유용

print('\n곡선의 경우에는 상관계수는 의미 없다. /--------------------')
m = [-3,-2,-1,0,1,2,3]
n = [9,4,1,0,1,4,9]
plt.plot(m,n)
plt.show()
print('m,n의 공분산:', np.cov(m,n)[0,1])         # m,n의 공분산: 0.0
print('m,n의 상관계수:', np.corrcoef(m,n)[0,1])   # m,n의 상관계수: 0.0