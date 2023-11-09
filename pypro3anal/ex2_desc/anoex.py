# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# H0 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 없다.
# H1 : 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 있다.

import pandas as pd
import numpy as np

data = pd.DataFrame({
    'kind': [1, 2, 3, 4, 2, 1, 3, 4, 2, 1, 2, 3, 4, 1, 2, 1, 1, 3, 4, 2],
    'quantity': [64, 72, 68, 77, 56, np.nan, 95, 78, 55, 91, 63, 49, 70, 80, 90, 33, 44, 55, 66, 77]
})
print(data)
df = data.fillna(data.mean())
print(df)
x1=df[df.kind == 1].quantity
x2=df[df.kind == 2].quantity
x3=df[df.kind == 3].quantity
x4=df[df.kind == 4].quantity
print(x1, np.mean(x1))
print(x2, np.mean(x2))
print(x3, np.mean(x3))
print(x4, np.mean(x4))

# 등분산성
print(stats.bartlett(x1,x2,x3,x4).pvalue) # 0.19342011099507922
# pvalue값이 0.05보다 크므로 귀무가설 채택
print('등분산성을 만족하지 않았으므로 welch_anova')
from pingouin import welch_anova
print(welch_anova(data=df, dv='quantity', between='kind'))
# 0.745712 > 0.05이므로 귀무가설 채택

# 정규성
print(stats.shapiro(x1).pvalue)
print(stats.shapiro(x2).pvalue)
print(stats.shapiro(x3).pvalue)
print(stats.shapiro(x4).pvalue)
# 4개의 변수 모두 pvalueㄱ밧이 0.05보다 크므로 귀무가설 채택

print('정규성을 만족하지 않았으므로 Kruskal-Walis test')
print(stats.kruskal(x1,x2,x3,x4))
# KruskalResult(statistic=0.9049322289156589, pvalue=0.8242373884048854) 귀무가설 채택

print('사후검정')
from statsmodels.stats.multicomp import pairwise_tukeyhsd
postHoc = pairwise_tukeyhsd(df['quantity'], df['kind'], alpha=0.05)
print(postHoc)
postHoc.plot_simultaneous()
plt.show()


# [ANOVA 예제 2]
# DB에 저장된 buser와 jikwon 테이블을 이용하여 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오.
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.
import MySQLdb
import pickle
import sys
try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 :',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql="""
        select buser_name, jikwon_name, jikwon_pay from jikwon inner join buser on buser_num=buser_no
    """
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['부서명', '직원명','연봉'])
    data = data.fillna(data)
    print(data)
    
    총무부 = data[data['부서명'] == '총무부']
    영업부 = data[data['부서명'] == '영업부']
    전산부 = data[data['부서명'] == '전산부']
    관리부 = data[data['부서명'] == '관리부']
    # print(총무부)
    x1=총무부['연봉']
    x2=영업부['연봉']
    x3=전산부['연봉']
    x4=관리부['연봉']
    print(x1, np.mean(x1))
    print(x2, np.mean(x2))
    print(x3, np.mean(x3))
    print(x4, np.mean(x4))
    # H0 : 부서들 간 직원 연봉의 평균에 차이가 없다.
    # H1 : 부서들 간 직원 연봉의 평균에 차이가 있다.
    
    # 등분산성
    print(stats.bartlett(x1, x2, x3, x4).pvalue)
    # 0.6290955395410989이므로 0.05보다 크기 때문에 귀무가설 채택
    
    # 정규성
    print(stats.ks_2samp(x1, x2).pvalue, stats.ks_2samp(x1, x3).pvalue, stats.ks_2samp(x1, x4).pvalue,
          stats.ks_2samp(x2,x3).pvalue ,stats.ks_2samp(x2,x4).pvalue, stats.ks_2samp(x3,x4).pvalue)
    # 0.33577439072795107 0.5751748251748252 0.5363636363636364 0.33577439072795107 0.6406593406593406 0.5363636363636364
    # pvalue값들 모두 0.05보다 크기에 귀무가설 채택, 부서들 간 직원 연봉의 평균에 차이가 없다.
    print(stats.f_oneway(x1, x2, x3, x4))
    # F_onewayResult(statistic=0.41244077160708414, pvalue=0.7454421884076983) pvalue > 0.05이므로 귀무가설 채택
except Exception as e:
    print('처리 오류 : ',e)
finally:
    cursor.close()
    conn.close()