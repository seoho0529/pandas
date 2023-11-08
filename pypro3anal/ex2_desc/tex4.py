# 독립표본 t검정
# 어느 음식점의 매출 데이터와 날씨 데이터, 두 개의 파일을 이용해 강수 여부에 따른 음식점 매출액 평규균의 차이를 검정
# 귀무가설 : 강수 여부에 따른 음식점 매출액 평균의 차이는 없다.
# 대립가설 : 강수 여부에 따른 음식점 매출액 평균의 차이는 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 데이터
sales_data = pd.read_csv('../testdata/tsales.csv', dtype={'YMD':'object'}) # YMD를 문자형으로 바꿈
print(sales_data.head(3))  # YMD : 20190514
print(sales_data.info())

# 날씨 데이터
wt_data = pd.read_csv('../testdata/tweather.csv')
print(wt_data.head(3))  # tm : 2018-06-01
print(wt_data.info())
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-','')) # 함수를 실행하는 함수 map
print(wt_data.head(3))  # tm : 20180601

# merge
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm') # 매출액 날짜가 중요해서 left,, & left outer join
print(frame.head(3))
print(frame.tail(3), frame.shape)
print(frame.columns) # ['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn','maxWs', 'avgWs', 'ddMes']

data = frame.iloc[:, [0,1,7,8]] # 행은다 갖고와야해서 컬럼만 지정  YMD(날짜) AMT(매출액)  maxTa(최고기온)  sumRn(강수량)
print(data.head(3))
print(data.isnull().sum())

print('\n------two-sample t-test-------------------------------------------')
# print(data['sumRn'] > 0)
# data['data_yn'] = (data['sumRn'] > 0).astype(int)
# print(data.head(3))

# print(True * 1, False * 1)
data['data_yn'] = (data.loc[:,('sumRn')] > 0) * 1 # 위 문장과 같음
print(data.head(3))

# 매출액 시각화
import matplotlib.pyplot as plt

sp = np.array(data.iloc[:, [1,4]]) # 매출액, 비가 온다 안온다(0,1)
# print(sp)
tg1 = sp[sp[:,1] == 0, 0]  # 집단1 : 비 안올 때 매출액
tg2 = sp[sp[:,1] == 1, 0]  # 집단2 : 비 올 때 매출액
print(tg1[:3])
print(tg2[:3])

# plt.plot(tg1)
# plt.show()
# plt.plot(tg2)
# plt.show()
plt.boxplot([tg1,tg2], notch=True, meanline=True, showmeans=True)
plt.show()
print(np.mean(tg1),' ', np.mean(tg2)) # 761040.25   757331.52  // 두 집단 평균 차이가 있는지 없는지,,

# 정규성
print(len(tg1), len(tg2))
print(stats.shapiro(tg1).pvalue)
print(stats.shapiro(tg2).pvalue)

# 등분산성
print(stats.levene(tg1, tg2).pvalue)  # 0.7123452333011173

print(stats.ttest_ind(tg1, tg2, equal_var=True)) # TtestResult(statistic=0.10109828602924716, pvalue=0.919534587722196, df=326.0)
# 해석 : pvalue=0.9195 > 0.05 이므로 귀무가설 채택 -> 귀무가설 기각 실패