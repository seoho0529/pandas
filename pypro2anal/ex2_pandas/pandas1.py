# pandas : 고수준의 자료구조(Series, DataFrame)를 지원.
# 데이터 관리, 축약연산, 시게열 처리, 누락 데이터 처리, SQL 실행, SpreadSheet, 데이터 재배치, 시각화 등의 작업 수행이 가능하다.
# data munging, wrangling : 원 재료를 새로운 형태의 데이터로 전환하는 매핑 작업을 효율적으로 수행하도록 도와줌(by pandas)

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

# Series : 일련의 데이터(객체)를 담을 수 있는 1차원 배열과 같은 구조로 색인을 갖는다. 인덱싱된 데이터의 1차원 배열이다!\
obj = pd.Series([3,7,-5,4])   # 리스트
# obj = pd.Series((3,7,-5,5)) # 튜플 순서o, 수정x
#obj = pd.Series({3,7,-5,4})   # set

print(obj, type(obj))
# index 에 있어서 numpy는 묵시적 pandas는 명시적
obj2 = pd.Series([3,7,-5,5], index = ['a','b','c','d'])
print(obj2, type(obj2))
print(obj2.sum(), sum(obj2), np.sum(obj2))
print(obj2.values)  # [ 3  7 -5  5]
print(obj2.index)

# 인덱싱/슬라이싱 
print(obj2['a'])  # 값 리턴
print(obj2[['a']])# 인덱스 값 리턴
print(obj2[['a','b']])
print(obj2['a':'c'])

print(obj2[2])
print(obj2[1:4])
print(obj2[[2,1]])
print(obj2 > 0)
print('a' in obj2)

print('\ndict type : Series 객체로 처리')
names={'mouse':5000, 'keyboard':25000, 'monitor':550000}
print(names)
obj3 = Series(names)
print(obj3, type(obj3))
obj3.index = ['마우스','키보드','모니터']
print(obj3)
print(obj3[0], ' ', obj3['마우스'])

obj3.name='상품가격'  # 객체에 이름을 부여할 수 있다.
print(obj3)

print("\nDataFrame : 표 모양 - Series가 여러 개 합쳐진 형태")
df = DataFrame(obj3)
print(df, type(df))

data = {
    'irum':['홍길동', '공기밥', '김밥', '주먹밥'],
    'juso':('역삼동', '신당동', '신사동', '신당동'),
    'nai':[23,25,27,29],
}
print(data, type(data))
# DataFrame을 사용함으로써 dict타입이 표의 형태를 이룸
df2 = pd.DataFrame(data)
print(df2, type(df2))
print(df2['irum'], type(df2['irum']))  # Series
print(df2.irum, type(df2.irum))
print()

print(DataFrame(data, columns=['juso','irum','nai']))  # 컬럼의 순서도 바꿀 수 있다

print('data에 없는 값을 주면 NaN으로 채움')
df3 = DataFrame(data, columns=['irum','juso','nai','tel'],\
                index=['a','b','c','d'])
print(df3)

df3['tel'] = '111-1111'
print(df3)

tvalue = Series(['222-2222','333-3333','444-4444'], index=['b','c','d'])  # 인덱스를 지정해줘야 NaN값을 대체해줄 수 있음
df3['tel'] = tvalue
print(df3)

print('전치')
print(df3.T)

print(df3.values)
print(df3.values[0,1]) # 0행 1열 ,, 인덱싱 (데이터 값 하나만 찍기)
print(df3.values[0:2]) # 슬라이싱(더미로 찍기)

print('행 또는 열 삭제')
df4 = df3.drop(['d'])
# df4 = df3.drop(['d'], axis=0)  # 위와 같은 뜻,, 행 삭제
print(df4)
df4 = df3.drop('tel', axis=1)  # 열 삭제
print(df4)

# 정렬
print(df4.sort_index(axis=0, ascending=False))  # 행 단위
print()
print(df4.sort_index(axis=1, ascending=True))   # 열 단위,, 오름차순 : irum, juso, nai | 내림차순 : nai, juso, irum
print()
print(df4.rank(axis=0))  # 사전 순 순위를 매김

print()
counts = df4['juso'].value_counts()
print('열 값 개수 : ', counts)

print('문자열 자르기')
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 21]
}
fr = DataFrame(data)
print(fr)
# 리스트나 튜플로 자류구조를 변경하여 문자열을 자를 수 있다.
result1 = Series([x.split()[0] for x in fr.juso])
print(result1)
print(result1.value_counts())
result2 = Series((x.split()[1] for x in fr.juso))
print(result2)