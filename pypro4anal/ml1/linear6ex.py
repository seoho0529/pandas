# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api
import numpy as np

df = pd.read_csv('../testdata/carseats.csv')
print(df.head(3), df.shape)
print(df.info())
df['Price'] = df['Price'].astype(float)
df['Advertising'] = df['Advertising'].astype(float)
print(df.dtypes)
print(df['Price'].unique())
print(df['Advertising'].unique())
print(df.loc[:,['Sales','CompPrice','Income','Advertising','Population','Price','Age','Education']].corr())
# Sales 와 Price가 상관계수가 제일 높기 떄문에 두 변수 선택, 독립:Price 종속:Sales

# 단순선형회귀 모델 작성
result = smf.ols('Sales ~ Price', data=df).fit()
print(result.summary())
# Prob (F-statistic):7.62e-21 < 0.05 이므로 유의한 모델이라 판단할 수 있으며 R-squared:0.198의 모형 설명력을 가지고 있다.
# 또한 Price 변수의 pvalue는 0.05보다 작기 때문에 종속변수인 Sales에 유의한 영향을 미칠 수 있는 변수라고 볼 수 있다.

# 모델 검정
pred = result.predict(df[:10])
print('실제값 : ', df.Sales[:10].values)
print('예측값 : ', pred[:10].values)


# 잔차항 구하기
fitted = result.predict(df.iloc[:, [0, -6]])
print(fitted)
residual = df['Sales'] - fitted
print(residual)

# 선형성
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0,0])
plt.show()
# 예측값과 잔차가 직성상에 있는 것 처럼 보이기 때문에 선형성 만족으로 본다.


# 정규성
import scipy.stats
ssz = scipy.stats.zscore(residual)
(x,y),_= scipy.stats.probplot(ssz)
sns.scatterplot(x=x, y=y)
plt.plot([-3,3],[-3,3], '--', color='blue')
plt.show()
# 정규성 또한 직선을 따라가고 있기 때문에 정규성 만족으로 본다.

# 독립성
# Durbin-Watson:1.892 으로 2에 가깝게 나왔기 때문에 자기상관이 없으며 독립성을 띈다고 볼 수 있다.

# 등분산성
sns.regplot(x=fitted, y=np.sqrt(np.abs(ssz)), lowess=True, line_kws={'color':'red'})
plt.show()
# 등분산성은 곡선을 그리거나 이분산성의 형태를 띄지 않고 추체선이 뜨기 때문에 등분산성 또한 만족으로 판단할 수 있다.

# 독립변수가 한개이기 때문에 다중공선성은 하지않음

#  Cooks distance로 이상치 확인하기
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(result).cooks_distance # 극단값을 나타내는 지표를 반환
print(cd.sort_values(ascending=False).head())
 
statsmodels.api.graphics.influence_plot(result, criterion='cooks')
plt.show()


# 이상치 boxplot으로 확인하기
plt.figure(figsize = (10, 5))
plt.boxplot([df['Sales'], df['Price']])
plt.show()

# 회귀식 : Sales = -0.0531 * Price + 13.6419
# 예측1 : 새로운 Price 값으로 Sales를 추정
x_new = pd.DataFrame({'Price':[10000, 20000, 30000]})
new_pred = result.predict(x_new)
print('Sales 추정값 : ', new_pred.values)

print()
print('~~~~~~~~~~~~~~~'*10)
print()

# 다중선형회귀 모델 작성
result2 = smf.ols('Sales ~ Advertising + Price', data=df).fit()
print(result2.summary())
# Prob (F-statistic):2.87e-29 < 0.05 이므로 유의한 모델이라 판단할 수 있으며 Adj.R-squared:0.278의 모형 설명력을 가지고 있다.
# Advertising, Price 두 변수 모두 pvalue가 0.05보다 작기 때문에 종속변수인 Sales에 유의한 영향을 미칠 수 있는 변수라고 볼 수 있다.

print('잔차항 구하기')
fitted2 = result2.predict(df.iloc[:, [0,3,5]])
# print(fitted2)
residual2 = df['Sales'] - fitted2

# 선형성
sns.regplot(x=fitted2, y=residual2, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='blue')
plt.show()  # 직성상에 위치하기 떄문에 선형성 만족

# 정규성
import scipy.stats
ssz2 = scipy.stats.zscore(residual2)
(x, y), _ = scipy.stats.probplot(ssz2)
sns.scatterplot(x=x, y=y)
plt.plot([-3,3],[-3,3], '--', color='blue')
plt.show()  # 직선 상에 위치하기 떄문에 정규성 만족
print('정규성 : ', scipy.stats.shapiro(residual2)) # pvalue=0.32682064175605774 > 0.05 보다 크기 때문에 수치로도 정규성을 만족한다.

# 독립성
print('Durbin-Watson : 1.964') # 2에 가깝기 때문에 독립성 만족
print(df.dtypes)

# 다중공선성
from statsmodels.stats.outliers_influence import variance_inflation_factor
df2 = df[['Price','Advertising']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(df2.values,i)for i in range(df2.shape[1])] 
print(vifdf)
# vif 지수가 10보다 작기 때문에 다중공선성을 만족한다.

# 모델 검증이 끝난 경우 모델을 저장
# 방법1 - 모델 저장 후 불러오기
# import pickle # 객체로 파일로 저장가능한 pickle
# with open('linear6ex_m.model','wb') as obj:
#     pickle.dump('result2', obj)
#
# with open('linear6ex_m.model','rb') as obj: 
#     mymodel = pickle.load(obj)
    
# 방법2 - 메모리가 더 절약되는 방법2를 더 선호
import joblib
joblib.dump(result2, 'linear6ex_m.model')

mymodel = joblib.load('linear6ex_m.model')  
    
# 예측2 : 새로운 Advertising, Price 값으로 Sales를 추정(구간추정)
# 다중회귀식 : Sales = 0.1231 * Advertising + (-0.0546) * Price + 13.0034
x_new2 = pd.DataFrame({'Advertising':[1000, 2000, 3000], 'Price':[100, 200, 300]})
new_pred2 = mymodel.predict(x_new2)
print('sales 추정값 : ', new_pred2.values)
"""
이렇게 하니까 안됨
# 다중공선성 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vifdf = pd.DataFrame()
vifdf['Variable'] = df.columns[[3, 5]]  # 첫 번째 열은 종속변수이므로 제외
vifdf['VIF'] = [variance_inflation_factor(df.values, i) for i in range(1, df.shape[1])]
print(vifdf)
"""



