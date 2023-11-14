# LinearRegression 클래스를 사용해 선형회귀모델을 작성

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 편차가 큰 표본 데이터를 생성
sample_size = 100

np.random.seed(1)
x = np.random.normal(0, 10, sample_size) # 평균0, 표준편차10
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:10])
print(y[:10])  # x,y 모두 정규분포를 따르기 때문에 상관계수 또한 높을 거라 예측
print('상관계수 : ', np.corrcoef(x,y))
# 만약 이상치가 있다고 의심되면 RobustScaler

# 정규화
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled[:10].flatten()) # flatten() : 차원축소
# log를 사용하면 데이터의 편차가 줄어든다.

# 시각화
# plt.scatter(x_scaled, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y) # feature(독립):x_scaled, label(종속):y
print(model)
y_pred = model.predict(x_scaled)

print('실제값 : ', y[:10])
print('예측값 : ', y_pred[:10])

# 모델 성능 확인
# print(model.summary())  # LinearRegression는 summary함수를 지원x, ols가 지원함

def regScoreFunc(y_true, y_pred): # summary함수가 안되서 이런식으로 직접 값 출력
    print('r2_score(결정계수, 설명력) : {}'.format(r2_score(y_true, y_pred)))
    print('explained_variance_score(설명분산점수) : {}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(MSE, 평균제곱오차) : {}'.format(mean_squared_error(y_true, y_pred)))


regScoreFunc(y, y_pred)
# r2_score(결정계수, 설명력) : 0.9987875127274646
# explained_variance_score(설명분산점수) : 0.9987875127274646  
# 결정계수와 설명분산점수의 결과의 차이(편향)가 크다면 모델학습이 잘못됐다고 판단
# mean_squared_error(MSE, 평균제곱오차, SSE와 같음) : 86.14795101998743
# MSE는 0에 가깝게 작아질수록 좋은 모델을 의미한다. 결정계수와 MSE값은 반비례 관계

print('-----------'*10)
# 분산이 크게 다른 표본 데이터를 생성 (분산이 크다.-> 밀도가 떨어진다.-> 많이 떨어져있다.)
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30

print(x[:10])
print(y[:10])
print('상관계수 : ', np.corrcoef(x,y))  # 0.00401167

# 정규화
x_scaled2 = scaler.fit_transform(x.reshape(-1,1))
print(x_scaled2[:10].flatten())

# 모델 작성
model2 = LinearRegression().fit(x_scaled2, y) # feature(독립):x_scaled, label(종속):y
print(model2)
y_pred2 = model2.predict(x_scaled2)

print('실제값 : ', y[:10])
print('예측값 : ', y_pred2[:10])

regScoreFunc(y, y_pred2)
# r2_score(결정계수, 설명력) : 1.6093526521765433e-05
# explained_variance_score(설명분산점수) : 1.6093526521765433e-05
# mean_squared_error(MSE, 평균제곱오차) : 282457.9703485092
