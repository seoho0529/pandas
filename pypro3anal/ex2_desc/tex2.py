# 추론 통계 분석 중 가설검정 : 독립표본 t-검정(independent two-sample t-test)
# 비교 집단이 두 개인 경우에 평균이 단순히 유의한 차이가 있는지를 검정
# ex) 오리온과 롯데에서 생산된 초코파이의 무게가 같은가?

import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.
# 실습) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정. 남녀의 시험 평균이 우연히 같을 확률은 얼마일까?
# H0 : 두 집단 간 파이썬 시험의 평균에 차이가 없다.
# H1 : 두 집단 간 파이썬 시험의 평균에 차이가 있다.
male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
print(np.mean(male), np.mean(female)) # 83.8 72.24
# equal_var=True : 등분산성 , alternative='two-sided' : 양측검정
two_sample = stats.ttest_ind(male, female, equal_var=True, alternative='two-sided')
print(two_sample) # TtestResult(statistic=1.233193127514512, pvalue=0.2525076844853278, df=8.0)
# 해석 : pvalue=0.2525 > 0.05 이므로 귀무가설을 채택한다.

# 참고 : Effect-size(효과크기) - 두 집단 평균 차이를 물리적으로 세 등급으로 분리해 표현
# 효과크기(Cohen's d) = 두 표본 집단의 평균 차이 / 추정된 표준편차
imsi = male + female
print(imsi)
print((np.mean(male) - np.mean(female)) / np.std(imsi)) # 0.799이므로 효과크기가 커서 두 그룹의 평균에 차이가 크다고 할 수 있다.

# 실습) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'
# H0 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# H1 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.
data = pd.read_csv('../testdata/two_sample.csv')
print(data.head(3), data.shape) # (50, 5)
print(data.isnull().sum()) # score에서 null이 2개임을 확인

ms = data[['method','score']]
print(ms.head(2))
print(ms['method'].unique()) # [1 2]

print('교육방법별 데이터 추출')
m1 = ms[ms['method'] == 1]
m2 = ms[ms['method'] == 2]
print(m1.head(2))
print(m2.head(2))

print('교육방법에서 점수 추출')
score1 = m1['score']
score2 = m2['score']
print(score1.isnull().sum())  # 0
print(score2.isnull().sum())  # 2   결측값은 제거 or 0 or 평균으로 대체가능
# score2 = score2.fillna(0)  결측치를 0으로 채우기
score1 = score1.fillna(score1.mean()) # 결측치를 평균으로 채우기
score2 = score2.fillna(score2.mean()) # 결측치를 평균으로 채우기
print(score2)

print('정규성 확인')
import seaborn as sns
sns.histplot(score1, kde=True, color='r')
sns.histplot(score2, kde=True, color='r')
plt.show()

print(stats.shapiro(score1).pvalue) # 0.36798644065856934 > 0.05 이므로 정규성 만족
print(stats.shapiro(score2).pvalue) # 0.6714232563972473 > 0.05 이므로 정규성 만족

print('등분산성 확인')
print(stats.levene(score1, score2).pvalue)   # 모수검정일때 fligner     0.456 > 0.05 이므로 등분산성 만족
print(stats.fligner(score1, score2).pvalue)  # 모수검정일때 fligner     0.443 > 0.05 이므로 등분산성 만족
print(stats.bartlett(score1, score2).pvalue) # 비모수일때 bartlett 사용  0.267 > 0.05 이므로 등분산성 만족

result = stats.ttest_ind(score1, score2)
print('t-value : %.5f, p-value : %.5f'%result) # t-value : -0.19649, p-value : 0.84505
print(np.mean(score1), np.mean(score2)) # 5.19 5.25
# 판정 : p-value : 0.845 > 0.05 이므로 귀무가설을 채택한다. 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.

print('등분산성을 만족한 경우 : ', stats.ttest_ind(score1, score2).pvalue)
print('등분산성을 만족한 경우 : ', stats.ttest_ind(score1, score2, equal_var=True).pvalue)
print('등분산성을 만족하지 못한 경우 : ', stats.ttest_ind(score1, score2, equal_var=False).pvalue)
# 등분산성을 만족하지 못한 경우 welchi test도 가능

print()
print('정규성을 만족한 경우 : ', stats.ttest_ind(score1, score2).pvalue)
# print('정규성을 만족하지 못한 경우 : ', stats.wilcoxon(score1, score2).pvalue)   # 두 집단의 크기가 같은 경우,, score1,2크기가 서로 달라서 에러남
print('정규성을 만족하지 못한 경우 : ', stats.mannwhitneyu(score1, score2).pvalue) # 두 집단의 크기가 다른 경우
