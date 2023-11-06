# 1) human.csv 파일을 읽어 아래와 같이 처리하시오.
#      - Group이 NA인 행은 삭제
#      - Career, Score 칼럼을 추출하여 데이터프레임을 작성
#      - Career, Score 칼럼의 평균계산
import pandas as pd
"""
df=pd.read_csv('..\\testdata\\human.csv')
print(df)


df.columns = df.columns.str.strip()
print(df)
df = df[df['Group'].str.strip() != 'NA']
print(df)

df=df[['Career','Score']]
print(df)
carmean=df['Career'].mean()
print(carmean)
"""

 # 2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
 #     - 파일 정보 확인
 #     - 앞에서 3개의 행만 출력
 #     - 요약 통계량 보기
 #     - 흡연자, 비흡연자 수를 계산  : value_counts()
 #     - 요일을 가진 칼럼의 유일한 값 출력  : unique()
 #          결과 : ['Sun' 'Sat' 'Thur' 'Fri']
df=pd.read_csv('../testdata/tips.csv')
print(df.head(3))
print(df.describe())
counts = df['smoker'].value_counts()
print(counts)
unique_values = df['day'].unique()
print(unique_values)