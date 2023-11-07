# 추론 통계 분석 중 가설검정 : 단일표본 t-검정(one-sample t-test)
# 정규분포의(모집단) 표본에 대해 기댓값을 조사(평균차이 사용)하는 검정방법
# ex) 새우깡 과자 무게가 진짜 120g이 맞는가?

# 실습1) 어느 남성 집단의 평균키 검정
# 귀무가설 : 남성의 평균키는 177.0 (모집단의 평균)이다. (표본평균과 모평균이 같다)
# 대립가설 : 남성의 평균키는 177.0 (모집단의 평균)이 아니다. (표본평균과 모평균이 같지 않다)
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
# plt.boxplot(one_sample)
# plt.xlabel('data')
# plt.ylabel('height')
# plt.grid()
# plt.show()
print(np.array(one_sample).mean())  # 176.21999
print(np.array(one_sample).mean() - 177.0)  # -0.78
print('정규성 확인 : ', stats.shapiro(one_sample)) # 정규성 확인 :  pvalue=0.5400515794754028 > 0.05 이므로 정규성 만족
result = stats.ttest_1samp(one_sample, popmean=177.0)
print('result : ',result) # TtestResult(statistic=-0.22139444579394396, pvalue=0.8356282194243566, df=4)
print('statistic(t값):%.5f, pvalue:%.5f'%result)
# pvalue:0.83563 > 0.05 이므로 귀무가설을 채택한다. 즉, 수집된 one_sample 자료는 우연히 발생된 것이라 할 수 있다.

print('---'*10)
# 실습 예제 2)
# A중학교 1학년 1반 학생들의 시험결과가 담긴 파일을 읽어 처리 (국어 점수 평균검정) student.csv
# H0 : 학생들의 국어 점수 평균은 80.0이다.
# H1 : 학생들의 국어 점수 평균은 80.0이 아니다.

data = pd.read_csv('../testdata/student.csv')
print(data.head(3))
print(data.describe())
print(np.mean(data['국어'])) # data.국어 // 72.9 와 80.0
result2 = stats.ttest_1samp(data['국어'], popmean=80.0)
print('result2 : ',result2) # result2 :  TtestResult(statistic=-1.3321801667713216, pvalue=0.19856051824785262, df=19)
print('statistic(t값):%.5f, pvalue:%.5f'%result2)
# pvalue:0.19856 > 0.05 이므로 귀무가설을 채택한다. 즉, 학생들의 국어 점수 평균은 80.0이라 할 수 있다. 

print('---'*10)
# 실습 예제 3)
# 여아 신생아 몸무게의 평균 검정 수행 babyboom.csv
# 여아 신생아의 몸무게는 평균이 2800(g)으로 알려져 왔으나 이보다 더 크다는 주장이 나왔다.
# 표본으로 여아 18명을 뽑아 체중을 측정하였다고 할 때 새로운 주장이 맞는지 검정해 보자.

# H0 : 여아 신생아의 몸무게는 평균이 2800(g)이다.
# H1 : 여아 신생아의 몸무게는 평균이 2800(g)보다 크다.
data2 = pd.read_csv('../testdata/babyboom.csv')
print(data2.head(3), len(data2)) # 44
fdata = data2[data2['gender'] == 1] # 여아들만 뽑기
print(fdata, len(fdata)) # 18
print(np.mean(fdata['weight'])) # 3132.44

# 정규성 확인 수치
print(stats.shapiro(fdata.iloc[:, 2])) # ShapiroResult(statistic=0.8702831268310547, pvalue=0.017984924837946892) < 0.05 이므로 정규성을 불만족한다.

# 정규성 확인 시각화1
stats.probplot(fdata.iloc[:, 2], plot = plt)   # Q-Q plot
#plt.show()  # 직선에 붙어야 정규성 만족

# 정규성 확인 시각화2 : histogram
sns.displot(fdata.iloc[:, 2], kde=True)
#plt.show()

result3 = stats.ttest_1samp(fdata['weight'], popmean=2800)
print('result3 : ',result3) # result3 :  TtestResult(statistic=2.233187669387536, pvalue=0.03926844173060218, df=17)
print('statistic(t값):%.5f, pvalue:%.5f'%result3) # statistic(t값):2.23319, pvalue:0.03927
# pvalue:0.03927 < 0.05 이므로 귀무가설을 기각한다. 즉, 여아 신생아의 몸무게는 평균이 2800(g)보다 크다.


# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.

# H0 : 영사기에 사용되는 구형 백열전구의 수명은 300시간이다.
# H1 : 영사기에 사용되는 구형 백열전구의 수명은 300시간이 아니다.
sample = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print('정규성 확인 : ', stats.shapiro(sample))# 정규성 확인 : ShapiroResult(statistic=0.9661144614219666, pvalue=0.8208622932434082)
# pvalue=0.8208 > 0.05 이므로 정규성을 만족한다.
result4 = stats.ttest_1samp(sample, popmean=300.0)
print('result4 : ',result4) # result4 : TtestResult(statistic=-1.556435658177089, pvalue=0.143606254517609, df=13)
print('statistic(t값):%.5f, pvalue:%.5f'%result4)
# pvalue:0.14361 > 0.05 이므로 귀무가설을 채택한다. 즉, 수집된 sample 자료는 우연히 발생된 것이라 할 수 있다.


# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다.
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
# 실습 파일 : one_sample.csv
# 참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")
ndata = pd.read_csv('../testdata/one_sample.csv')
print(ndata)
# H0 : A회사에서 생상된 노트북 평균 사용 시간은 5.2시간이다.
# H1 : A회사에서 생상된 노트북 평균 사용 시간은 5.2시간이 아니다.
ndata['time'] = pd.to_numeric(ndata['time'].str.replace("     ", ""), errors='coerce')
print('정규성 확인 : ', stats.shapiro(ndata)) # 정규성 확인 : pvalue=1.0 > 0.05이므로 정규성 만족
result5 = stats.ttest_1samp(ndata['time'].dropna(), popmean=5.2)
print('result5 : ',result5) # result5 : TtestResult(statistic=-1.556435658177089, pvalue=0.143606254517609, df=13)
print('statistic(t값):%.5f, pvalue:%.5f'%result5)
# pvalue:0.00014 < 0.05  이므로 귀무가설 기각 대립가설을 채택한다. 즉, A회사에서 생상된 노트북 평균 사용 시간은 5.2시간이 아니다.


# [one-sample t 검정 : 문제3] 
# https://www.price.go.kr/tprice/portal/main/main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.
mdata = pd.read_csv('개인서비스지역별_동향(2023_09월)117_11시58분.csv', encoding='utf-8')
print(mdata)
# H0 : 전국 평균 미용 요금은 15000원이다.
# H1 : 전국 평균 미용 요금은 15000원이 아니다.
"""
mdata = mdata.dropna(axis=1)
mdata3 = mdata2.T
print(mdata3)
print(mdata3.iloc[2:,1])
"""
mdata = mdata.dropna(axis=1)
mdata = mdata.drop(['번호', '품목'], axis=1)
print(mdata.T)
print(np.mean(mdata.T.iloc[:,0]))

result6 = stats.ttest_1samp(mdata.iloc[0], popmean=15000 )
print(result6)
print('statistic(t값) : %.5f, pvalue:%.5f'%result6)
# pvalue:0.00001 < 0.05이므로 귀무가설 기각, 대립가설을 채택한다. 즉, 전국 평균 미용 요금은 15000원이 아니다.






