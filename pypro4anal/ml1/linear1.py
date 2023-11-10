# 단순선형회귀분석 모델 생성
# 독립변수(연속형), 종속변수(연속형) : 두 변수는 상관관계가 있어야 하고, 나아가서는 인과관계가 있다는 가정하에 작성을 한다.
# 회귀분석 각각의 데이터에 대한 잔차제곱합이 최소가 되는 수식을 도출해 내는 방법이다. 내부적으로 최소제곱법을 이용

import statsmodels.api as sm
import numpy as np
from sklearn.datasets import make_regression
from astropy.units import yyr


np.random.seed(12)

# 모델 생성 맛보기
print('방법1 : make_regression을 사용. 모델 생성이 안됨')
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True) # coef : 선형회귀계수,, 표본수50, 독립1
print('x, y, coef : ', x, y, coef)  # 회귀식 y = wx + b   
# 랜덤하게 x값(독립변수)을 생성[[-1.70073563] [-0.67794537] [ 0.31866529] ...
# 생성된 x값(독립변수)에 y값을 제시 [ -52.17214291   39.34130801  128.51235594] ...
# 학습 후 수식이 완성(모델) y = wx + b  => 예측값y = 89.47430739278907 * x + 100

pred_y = 89.47430739278907 * -1.70073563 + 100  # 작성된 모델로 x에 대한 예측값 y를 출력
print('y의 실제값은 ', -52.17214291)
print('x값 -1.70073563에 대한 예측값 y는 ', pred_y) # x값 -1.70073563에 대한 예측값 y는  -52.17214255248879

new_pred_y = 89.47430739278907 * 1234.5678 + 100
print('내가 궁금한 1234.5678에 대한 예측값은 ', new_pred_y)  # 내가 궁금한 1234.5678에 대한 예측값은 110562.09883443934

print('==========='*10)

xx = x
yy = y

print('방법2 : LinearRegression을 사용. 모델 생성 됨')
from sklearn.linear_model import LinearRegression
model = LinearRegression()  # LinearRegression 객체 생성
fit_model = model.fit(xx, yy)     # fit() : 최적의 추세선 찾기
print('기울기(slope) : ',fit_model.coef_)      # 기울기(slope) :  [89.47430739]
print('절편(bias) : ',fit_model.intercept_)    # 절편(bias) :  100.0

# 예측값 확인
print(xx[[0]])  # [[-1.70073563]] --> 2차원인 이유 : 학습할때 2차원으로 해서 fit(xx,yy)
y_new = fit_model.predict(xx[[0]])  # 예측값을 위해 predict(),, 사이킷런은 학습을 2차원으로 하기 때문에 [[]]

print('y_new : ',y_new)  # y_new :  [-52.17214291]

y_new2 = fit_model.predict(xx[[12]])
print('정말 궁금한 새로운 x값에 대한 예측결과 y는 ',y_new2) # 정말 궁금한 새로운 x값에 대한 예측결과 y는  [7.88349776]

print('==========='*10)

print('방법3 : ols 사용. 모델 생성 됨')
import statsmodels.formula.api as smf
import pandas as pd
print(xx.shape)  # (50, 1)
x1 = xx.flatten() # 차원축소
print(x1.shape)  # (50,)
y1 = yy
print(y1.shape)

data = np.array([x1, y1])
df = pd.DataFrame(data.T)
df.columns=['x1','y1']
print(df.head(3), len(df))

model2 = smf.ols(formula='y1 ~ x1', data=df).fit()
#model2.fit()
print(model2.summary())  # OLS Regression Results 표 제공

# 예측값 확인
print(x1[:2])  # [-1.70073563 -0.67794537]
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]})
print(new_df)
#        x1
# 0 -1.700736
# 1 -0.677945
new_pred = model2.predict(new_df)
print('예측값 new_pred :',new_pred) # 0   -52.172143,  1    39.341308
print('실제값 :', df.y1[:2])        # 0   -52.172143,  1    39.341308


# 전혀 새로운 독립변수 x 값에 대한 예측값
new2_df = pd.DataFrame({'x1':[111, -6.12345]})
new2_pred = model2.predict(new2_df)
print('새로운 독립변수 x 값에 대한 예측값 new2_pred :',new2_pred)



