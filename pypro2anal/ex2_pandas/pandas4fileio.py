# pandas로 파일 읽기
import pandas as pd

# df = pd.read_csv('../testdata/ex1.csv')
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ex1.csv') # 웹에서 가져오기 - raw
print(df, type(df))  # <class 'pandas.core.frame.DataFrame'>
print(df.info())
print('---'*10)
df = pd.read_table('../testdata/ex1.csv', sep=',')  # 문자열로 읽기
print(df.info())
print(df)
print()
df = pd.read_csv('../testdata/ex2.csv', header=None) # header=None 사용하면 자동으로 컬럼이 생김
print(df)

df = pd.read_csv('../testdata/ex2.csv', header=None, names=['col1','col2'])  # 오른쪽부터 칼럼명을 채움
print(df)
print()

df = pd.read_csv('../testdata/ex2.csv', header=None, names=['a','b','c','d','msg'], index_col='msg')
print(df)
print()

# df=pd.read_csv('../testdata/ex3.txt')
df=pd.read_table('../testdata/ex3.txt', sep='\s')  # sep=' ' or sep='정규표현식'
# \s : space를 표현하며 공백 문자를 의미한다.
# \S : non space를 표현하며 공백 문자가 아닌 것을 의미한다.
print(df)
print(df.info())
print(df.describe())
print()
df=pd.read_table('../testdata/ex3.txt', sep='\s+', skiprows=(1,3)) # skiprows : 특정 행 제외, \s : 공간 여러개?
print(df)

print()
df = pd.read_fwf('../testdata/data_fwt.txt', widths=(10,3,5),
                header=None, names=('date','name','price'), encoding='utf-8') # 데이터가 다 붙어있기 때문에 일정한 간격(폭)으로 불러와야함
print(df)
print('---'*10)

# 대용량의 자료를 chunk(묶음) 단위로 할당해서 처리 가능
test = pd.read_csv('../testdata/data_csv2.csv', header=None, chunksize=3)  # 3행씩 끊어읽기
print(test)  # TextFileReader object(텍스트 파서 객체)

for p in test:
    #print(p)
    print(p.sort_values(by=2, ascending=True)) # price별 정렬, 오름차순(작은것부터 큰것) 정렬

print('--- DataFrame 저장 ---')
items = {'apple':{'count':10, 'price':1500}, 'orange':{'count':5, 'price':1000}}
df = pd.DataFrame(items)
print(df)
# print(df.to_html())
# print(df.to_json())
# print(df.to_clipboard())
# print(df.to_csv())
df.to_csv('test1.csv', sep=',')
df.to_csv('test2.csv', sep=',', index=False)
df.to_csv('test3.csv', sep=',', index=False, header=False)
