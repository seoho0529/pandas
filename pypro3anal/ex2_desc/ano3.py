# 일원분산분석 검정
# 어느 음식점의 매출 데이터와 날씨 데이터, 두 개의 파일을 이용해 최고 온도에 따른 음식점 매출액 평규균의 차이를 검정
# 귀무가설 : 온도에 따른 음식점 매출액 평균의 차이는 없다.
# 대립가설 : 온도에 따른 음식점 매출액 평균의 차이는 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

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
print(data.maxTa.describe())
# count    328.000000
# mean      18.597866
# std       10.163039
# min       -4.900000
# 25%        9.375000
# 50%       19.350000
# 75%       27.800000
# max       36.800000

# plt.boxplot(data.maxTa)
# plt.show()

# 온도를 임의로 추움, 보통, 더움 (0,1,2) 세 구간으로 나눠 그룹을 형성
data['Ta_gubun'] = pd.cut(data.maxTa, bins=[-5,8,24,37], labels=[0,1,2])
print(data.head(3))

print(data.corr()) # 상관계수로 관계 확인

# 세 그룹으로 데이터를 분리 : 등분산성, 정규성 만족 여부 확인 - 종속변수가 해당
x1 = np.array(data[data.Ta_gubun == 0].AMT)
x2 = np.array(data[data.Ta_gubun == 1].AMT)
x3 = np.array(data[data.Ta_gubun == 2].AMT)
print('1')
print(x1[:10], len(x1))
print(x2[:10])
print(x3[:10])

# 등분산성
print(stats.levene(x1,x2,x3).pvalue)  # 0.039 < 0.05 이므로 등분산성을 만족하지 못한다.

# 정규성
print(stats.ks_2samp(x1, x2).pvalue, stats.ks_2samp(x1, x3).pvalue, stats.ks_2samp(x2, x3).pvalue)
# 9.289383890229753e-09 4.9182870966559816e-27 1.4265159297542065e-13 모두 0.05보다 작으므로 정규성을 따르지 않는다.
# 원래 등분산성, 정규성 모두 불만족하면 데이터를 다시 가공해야 한다. 하지만 지금은,,,

# 온도별 매출액 평균
tempAmt = data.loc[:, ['AMT', 'Ta_gubun']]
print(tempAmt.groupby('Ta_gubun').mean())

print(pd.pivot_table(tempAmt, index=['Ta_gubun'], aggfunc='mean'))

tempAmtArr = np.array(tempAmt)
print(tempAmtArr)  # [[      0       2]   [  18000       1]
group1 = tempAmtArr[tempAmtArr[:,1]==0, 0]
group2 = tempAmtArr[tempAmtArr[:,1]==1, 0]
group3 = tempAmtArr[tempAmtArr[:,1]==2, 0]

# plt.boxplot([group1,group2,group3], meanline=True, showmeans=True)
# plt.show()
print()
print(stats.f_oneway(group1,group2,group3))
print('\n------one-way anova -------------------------------------------')
# F_onewayResult(statistic=99.1908012029983, pvalue=2.360737101089604e-34)
# 표본 통계량이 크기 때문에 pvalue가 작다는 것을 알 수 있으며 p값이 0.05보다 작기 때문에 대립가설을 채택한다.

print('정규성을 만족하지 않았으므로 Kruskal-Walis test')
print(stats.kruskal(group1,group2,group3)) # pvalue=1.5278142583114522e-29)--> 귀무가설 기각

print('등분산성을 만족하지 않았으므로 welch_anova')
# pip install pingouin
from pingouin import welch_anova
print(welch_anova(data=data, dv='AMT', between='Ta_gubun')) # 7.907874e-35

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
postHoc = pairwise_tukeyhsd(tempAmt['AMT'], tempAmt['Ta_gubun'], alpha=0.05)
print(postHoc)

postHoc.plot_simultaneous()
plt.show()
