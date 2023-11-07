import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('[two-sample t 검정 : 문제2]')
# 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
import random
# H0 : 남자와 여자는 혈관 내의 콜레스테롤 양에 차이가 없다.
# H1 : 남자와 여자는 혈관 내의 콜레스테롤 양에 차이가 있다.
male=[0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female=[1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

male_sample = random.sample(male, 15)
female_sample = random.sample(female, 15)
pair_sample = stats.ttest_ind(male_sample, female_sample)
print('t-value : %.5f, p-value : %.5f'%pair_sample) # t-value : -1.34406, p-value : 0.18972
# p-value : p-value : 0.18972 > 0.05 보다 크므로 귀무가설을 채택하며 남자와 여자는 혈관 내의 콜레스테롤 양에 차이가 없다.


print('[two-sample t 검정 : 문제3]')
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.
import MySQLdb
import pickle
import sys
try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()
    
try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = '''
        select buser_name, jikwon_pay from jikwon inner join buser on buser_num = buser_no
    '''
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=['부서명', '연봉'])
    jikwon = data.fillna(data)
    print(jikwon)
    총무부직원 = jikwon[jikwon['부서명']=='총무부']
    영업부직원 = jikwon[jikwon['부서명']=='영업부']
    print(총무부직원)
    print(영업부직원)
    # H0 : 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.
    # H1 : 총무부, 영업부 직원의 연봉의 평균에 차이가 존재한다.
    cpay=총무부직원['연봉']
    ypay=영업부직원['연봉']
    
    pay = stats.ttest_ind(cpay, ypay)
    print('t-value : %.5f, p-value : %.5f'%pay) # t-value : 0.45852, p-value : 0.65239
    # p-value : 0.65239 > 0.05 보다 크므로 귀무가설을 채택, 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하지 않는다.
except Exception as e:
    print('처리 오류 : ',e)
finally:
    cursor.close()
    conn.close()


print('[대응표본 t 검정 : 문제4]')
# 어느 학급의 교사는 매년 학기 내 치뤄지는 시험성적의 결과가 실력의 차이없이 비슷하게 유지되고 있다고 말하고 있다. 이 때, 올해의 해당 학급의 중간고사 성적과 기말고사 성적은 다음과 같다. 점수는 학생 번호 순으로 배열되어 있다.
#    중간 : 80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80
#    기말 : 90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95
# 그렇다면 이 학급의 학업능력이 변화했다고 이야기 할 수 있는가?
중간 = [80, 75, 85, 50, 60, 75, 45, 70, 90, 95, 85, 80]
기말 = [90, 70, 90, 65, 80, 85, 65, 75, 80, 90, 95, 95]
# H0 : 학급의 학업능력은 변화하지 않았다.
# H1 : 학급의 학업능력은 변화하였다.
print(np.mean(중간))
print(np.mean(기말))

pair_sample = stats.ttest_rel(중간, 기말)
print('t-value : %.5f, p-value : %.5f'%pair_sample) # t-value : -2.62811, p-value : 0.02349
# p-value : 0.02349 < 0.05 보다 작으므로 귀무가설 기각, 대립가설을 채택한다. 즉, 학급의 학업능력은 변화하였다.