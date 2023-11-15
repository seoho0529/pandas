# Logistic Linear Regression
# 선형회귀분석처럼 신뢰구간, 표준오차, p값 등이 제공되나 회귀계수의 결과를 해석하는 방법이 다르다.
# 독립변수 : 연속형, 종속변수 : 범주형, 이항분포를 따르며 출력값은 0 ~ 1 사이의 확률로 제공된다.
# 연속형 결과물 로짓(오즈비에 로그를 씌움)변환 후 시그모이드 함수를 통해 결과를 내보낸다.

import math
# sigmoid function 경험
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

print(sigmoidFunc(3))
print(sigmoidFunc(1))
print(sigmoidFunc(-2))
print(sigmoidFunc(-5))

print('mtcars dataset을 사용')
import statsmodels.api as sm

mtcarData = sm.datasets.get_rdataset('mtcars')
print(mtcarData.keys())
mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(2))
mtcar = mtcars.loc[:,['mpg','hp','am']]  # 연비(연속),마력수(연속),변속기
print(mtcar['am'].unique())   # [1 0]

# 연비와 마력수는 변속기에 영향을 주는가?
# 모델 작성 방법1 : logit() - confusion matrix 지원
import statsmodels.formula.api as smf
formula='am~hp+mpg'
model1 = smf.logit(formula=formula, data=mtcar).fit()
print(model1)
# Optimization terminated successfully.              - 오차를 최소한으로 했음
#          Current function value: 0.300509          - 목적 함수 값: 0.300509, 이 값이 낮을수록 최적화가 더 수렴하고 있다.
#          Iterations 9                              - 9번의 학습을 반복함
print(model1.summary())
# 각 변수들에 대한 pvalue는 나오지만, 모델에 대한 p값은 나오지 않음 --> 로지스틱 회귀분석

import numpy as np
pred = model1.predict(mtcar[:10])
print('예측값 : ', pred.values)
print('예측값 : ', np.around(pred.values))  # around()를 통해 0.5를 기준으로 작으면 0, 크면 1로 나옴
print('실제값 : ', mtcar['am'][:10].values) # mtcar의 am칼럼 10개

conf_tab = model1.pred_table()
print('confusion matrix : \n', conf_tab)  # confusion matrix : 혼동행렬
#     예측  0    1
# 실제 0  [[16.  3.]
#     1  [ 3. 10.]]  # 16,10은 예측맞음, 3, 3은 예측 틀린것

print()
print('분류 정확도 : ',(16+10) / len(mtcar))  # 0.8125이므로 모델의 분류 정확도는 약 81.25%이다.
print('분류 정확도 : ',(conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 예측맞은것에 대한 분류정확도

from sklearn.metrics import accuracy_score # 위처럼 행렬의 방식 말고 accuracy_score를 이용해 계산가능
pred2 = model1.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2))) # 실제값, 예측값,, 0.8125


# 모델 작성 방법2 : logit()
model2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
print(model2)
print(model2.summary())
glmPred = model2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glmPred).values)
print('glm 실제값 : ', mtcar['am'][:10].values)
glmPred2 = model2.predict(mtcar)
print('glm 분류 정확도 : ', accuracy_score(mtcar['am'], np.around(glmPred2))) # glm 분류 정확도 : 0.8125

print('새로운 값으로 분류 예측')
newdf = mtcar.iloc[:2].copy()  # 기존 자료 2행 읽어 값 수정 후 분류에 참여
# print(newdf)
newdf['mpg'] = [10,50]
newdf['hp'] = [100,150]
print(newdf)
new_pred = model2.predict(newdf)
print('new_pred : ', np.around(new_pred.values))
print('new_pred : ', np.rint(new_pred.values))   # rint()도 around()와 같은 결과를 낼 수 있음

print()
import pandas as pd
newdf2 = pd.DataFrame({'mpg':[10,35,50,5], 'hp':[80,100,125,50]})
new_pred2 = model2.predict(newdf2)
print('new_pred2 : ', np.around(new_pred2.values))


# 머신러닝의 포용성(inclusion, tolerance)  -  모델은 포용성이 있어야함 (다양한 데이터와 상황에 대응하는 모델 또는 알고리즘의 능력)
# 생성모델은 최적화와 일반화를 잘 융합 (두가지가 잘 융합된 생성모델은 다양한 상황에 대응이 가능한 매우 좋은 머신러닝 모델)
# 분류정확도가 100%인 경우 과적합 모델이므로 새로운 데이터에 대해 정확한 분류를 할 수 없다. (꼬리없는 동물)

