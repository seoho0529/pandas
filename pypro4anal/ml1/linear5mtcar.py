# 단순선형회귀 : mtcars dataset, ols()
# 상관관계가 약한 경우와 강한 경우를 나눠 분석 모델 작성 후 비교

# 과학적 추론방식은 크게 2가지로 분류된다.
# 귀납법 : 개별 사례를 수집해 일반적인 법칙을 생성
# 연역법 : 사실이나 가정에 근거해 논리적 추론에 의해 결론을 도출  - 딥러닝에선 연역법을 사용

import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3), mtcars.shape)  # (32, 11)
print(mtcars.columns)
print(mtcars.describe())
print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0,1])  # -0.7761683718265862
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0,1])  # -0.8676593765172279

# 시각화
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수')
# plt.ylabel('연비')
# slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)  # 최소제곱법 사용
# plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
# plt.show()

print('\n----- 단순선형회귀 -----')
result = smf.ols('mpg ~ hp', data=mtcars).fit()
print(result.summary()) 
# Prob (F-statistic): 1.79e-07 < 0.05 이므로 유의한 모델, R-squared:0.602의 설명력,
# hp의 표준오차도 0.010으로 0에 가깝게 나왔으며, P>|t| 또한 0.05보다 작으므로 hp 변수는 유의한 변수라고 볼 수 있다.
print('마력수:{}에 대한 연비 예측 결과:{}'.format(110, -0.0682 * 110 + 30.0989))
# 마력수:110에 대한 연비 예측 결과:22.5969
print('마력수:{}에 대한 연비 예측 결과:{}'.format(110, result.predict(pd.DataFrame({'hp':[110]})))) # 22.59375


print('\n----- 다중선형회귀 -----')
result2 = smf.ols('mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary()) 
print('마력수:{}, 차체무게:{}에 대한 연비 예측 결과:{}'.format(110,5, 
                                               result2.predict(pd.DataFrame({'hp':[110],'wt':[5]})))) 
# 마력수:110, 차체무게:5에 대한 연비 예측 결과:0    14.343092
# 만약 hp와 wt의 pvalue값이 0.05보다 크다면 분석에서 제외할 경우를 고려해보아야 한다.
