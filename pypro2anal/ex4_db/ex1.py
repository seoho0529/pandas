import MySQLdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rc('font',family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False

try:
    with open('mydb.dat', mode='rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    sql = """
        select jikwon_no, jikwon_name, buser_name, jikwon_pay, jikwon_jik,gogek_no,gogek_name,gogek_tel
        from jikwon inner join buser on buser_num=buser_no
        left join gogek on jikwon_no=gogek_damsano
    """
    cursor.execute(sql)
    # 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    df=pd.DataFrame(cursor.fetchall(), columns=['jikwon_no', 'jikwon_name', 'buser_name', 'jikwon_pay', 'jikwon_jik','gogek_no','gogek_name','gogek_tel'])
    print(df.head(3), len(df))
    
    # DataFrame의 자료를 파일로 저장
    import csv
    with open('jikdata2.csv', mode='w', encoding='utf-8') as fobj:
        writer=csv.writer(fobj)
        for r in cursor:
            writer.writerow(r)
    print()
    #  부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    print(df.groupby(['buser_name'])['jikwon_pay'].sum())
    print(df.groupby(['buser_name'])['jikwon_pay'].max())
    print(df.groupby(['buser_name'])['jikwon_pay'].min())
    
    print()
    # 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    jtab = pd.crosstab(df['buser_name'],df['jikwon_jik'], margins=True)
    print(jtab)
    
    # 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시    
    sql2 = """
          select jikwon_name,gogek_no,gogek_name,gogek_tel
          from gogek right join 
          jikwon on gogek_damsano=jikwon_no
      """
    cursor.execute(sql2)

    df2 = pd.DataFrame(cursor.fetchall(), columns=['직원명', '고객번호', '고객명', '고객전화'])
    # df2.fillna({'고객번호': '담당 고객 X', '고객명': '담당 고객 X', '고객전화': '담당 고객 X'}, inplace=True)
    df2.fillna('담당 고객 X', inplace=True)
    print(df2)
    print()
    # 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    buserpay=df.groupby(['buser_name'])['jikwon_pay'].mean()
    plt.bar(buserpay.index, buserpay)
    plt.title('부서명별 연봉')
    plt.xlabel('부서명')
    plt.ylabel('평균연봉')
    plt.show()
    
    # pivot_table을 사용하여 성별 연봉의 평균을 출력
    sql3 = """
            select jikwon_no,jikwon_name,buser_name,jikwon_pay,jikwon_jik,jikwon_gen
            from jikwon j join 
            buser b on b.buser_no=j.buser_num
        """
    cursor.execute(sql3)
    df3 = pd.DataFrame(cursor.fetchall(), columns=['사번', '이름', '부서명', '연봉', '직급', '성별'])
    print(df3.pivot_table(['연봉'], index=['성별'],aggfunc=np.mean))

    #성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
    pay=df3.groupby(['성별'])['연봉'].mean()
    plt.bar(pay.index, pay)
    plt.xlabel('성별')
    plt.ylabel('평균 연봉')
    plt.show()
    
    #부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
    gtab = pd.crosstab(df3['부서명'],df3['성별'], margins=True)
    print(gtab)
    
except Exception as e:
    print('처리 오류 : ',e)
finally:
    cursor.close()

