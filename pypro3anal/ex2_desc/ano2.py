# 일원분산분석
# 강남구에 있는 GS편의점 3개 지역 알바생의 급여에 대한 평균에 차이가 있는가?

# 귀무가설 : GS편의점 3개 지역 알바생의 급여에 대한 평균에 차이가 없다.
# 대립가설 : GS편의점 3개 지역 알바생의 급여에 대한 평균에 차이가 있다.

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import urllib.request
from statsmodels.stats.anova import anova_lm

url = 'https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/group3.txt'
# data = pd.read_csv(url, header=None)
# print(data.head(3), data.shape)  # (80, 4)
# data = data.values # DataFrame -> ndarray 로 변환,, 밑과 상동

data = np.genfromtxt(urllib.request.urlopen(url), delimiter=',')
print(data, type(data))  # <class 'numpy.ndarray'>
print(data.shape)        # (22,2)

gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(gr1, ' ', np.mean(gr1)) # 316.625
print(gr2, ' ', np.mean(gr2)) # 256.444
print(gr3, ' ', np.mean(gr3)) # 278.0

# 정규성
print(stats.shapiro(gr1).pvalue) # 정규성 만족
print(stats.shapiro(gr2).pvalue) 
print(stats.shapiro(gr3).pvalue) 

# 등분산성
print(stats.levene(gr1, gr2, gr3).pvalue)  # 0.0458
print(stats.bartlett(gr1, gr2, gr3).pvalue)# 0.3508

# 데이터 퍼짐정도 시각화
plt.boxplot([gr1, gr2, gr3], showmeans=True)
# plt.show()

# 일원분산분석 처리 방법1
df = pd.DataFrame(data, columns=['pay','group'])
print(df)
lmodel = ols('pay~C(group)', data=df).fit()  # 범주형임을 의미 : C('독립변수') : 변수가 범주형임을 표시,, # .fit() 모델
print(anova_lm(lmodel, type=1))  # 0.043589 < 0.05 이므로 귀무가설 기각

print()
# 일원분산분석 처리 방법2
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print('f_statistic:{}, p_value:{}'.format(f_statistic, p_value)) # f_statistic:3.7113359882669763, p_value:0.043589
# 방법1과 방법2의 결과가 동일함을 알 수 있으며 GS편의점 3개 지역 알바생의 급여에 대한 평균에 차이가 있다고 할 수 있다.

# 사후검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tkResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tkResult)

tkResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()