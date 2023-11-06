# 기술통계 분석
# 측정이나 실험에서 수집한 자료(data)의 정리, 요약, 해석, 표현 등을 통해 자료의 특성을 규명하는 통계적 방법이다.
# 도수분포표 : 특정 구간에 속하는 자료(변량 집합)의 수를 나타내는 표(빈도표)
import pandas as pd

frame = pd.read_csv("../testdata/ex_studentlist.csv")
print(frame.head(3))
print(frame['age'].mean())
print(frame['age'].var())
print(frame['age'].std())
print(frame.descrbe())
print(frame['bloodtype'].nunique()) # 데이터의 고유값의 총 수를 알고 싶을때 : nunique()

# 평균, 분산, 빈도수, 변수 간의 상관관계 ... 적당한 해석이 필요

# 도수분포표 : bloodtype
data1 = frame.groupby(['bloodtype'])['bloodtype'].count()
print(data1)  # 혈액형 별 인원수

data2 = pd.crosstab(index=frame['bloodtype'], columns='count')
print(data2)  # crosstab로 구한 혈액형 별 인원수

data3 = pd.crosstab(index=frame['bloodtype'], columns=frame['sex'])
print(data3)  # 성별, 혈액형 별 인원수

data4 = pd.crosstab(index=frame['bloodtype'], columns=frame['sex'], margins=True) # 소계
print(data4)  # 성별, 혈액형 별 인원수 및 소계

print(data4 / data4.loc['All','All'])  # 행열 비율

# 깔끔한 해석, 시각화 ...
