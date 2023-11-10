# 단순선형회귀분석 모델 작성 : ols() 함수 - OLS Regression Results 내용 알기
# 결정론적 선형회귀분석 방법 - 확률적 모형에 비해 불확실성이 덜하다.

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
import statsmodels.formula.api as smf
df = pd.read_csv("../testdata/drinking_water.csv")
print(df.head(3), df.shape)
print(df.corr(method='pearson'))   # 적절성 / 만족도 : 0.766853

# 독립변수(x, feature) : 적절성
# 종속변수(y, label) : 만족도
# 목적 : 주어진 feature와 결정적 기반에서 학습을 통해 최적의 회귀계(slope, bias)를 찾아내는 것

model = smf.ols(formula='만족도~적절성', data=df).fit()  # formula='종속~독립'
# print(model.summary())
print(model.params)
print(model.pvalues)

# 예측값
print(df.적절성[:5].values)
new_df = pd.DataFrame({'적절성':[4, 3, 4, 2, 2]})
new_pred = model.predict(new_df)
# R-squared : 0.588(약 58.8%) 설명력이 있는 모델로 검정
print('만족도 실제값 : ', df['만족도'][:5].values)
print('만족도 예측값 : ', new_pred.values)