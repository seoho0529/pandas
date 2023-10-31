# DataFrame : reshape, cut, merge, pivot
import numpy as np
import pandas as pd

df = pd.DataFrame(1000 + np.arange(6).reshape(2,3), index=['대전','서울'], columns=['2021','2022','2023'])
print(df)
print()
df_row = df.stack()  # 재구조화 '열 -> 행'으로 변환  (인덱스를 기준으로)
print(df_row)
df_col = df_row.unstack()  # '행 -> 열'로 변환 (인덱스를 컬럼으로)
print(df_col)

print('--- 범주화 : 연속형 자료를 범주형으로 변경 ---')
price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11]  # 구간 기준값
result_cut = pd.cut(price, cut)
print(result_cut)  # (a, b] : a 초과 b 이하 : a < x <= b
print(pd.value_counts(result_cut))
print()

datas = pd.Series(np.arange(1,1001))
print(datas.head(3))  # 앞에 3개
print(datas.tail(3))  # 뒤에 3개
result_cut2 = pd.qcut(datas, 3)   # datas 값을 3개 영역으로 범주화
print(result_cut2)
print(pd.value_counts(result_cut2))
print()
cut2 = [1, 500, 1000]
result_cut3 = pd.cut(datas, cut2)
print(result_cut3)
print(pd.value_counts(result_cut3))

print('그룹별 함수 수행 : agg, apply')
group_col = datas.groupby(result_cut2)  # groupby를 사용한 그룹별 함수 수행
print(group_col.agg(['count','mean','std','max']))  # 여기서 ''count','mean','std','max'들은 함수임
print('---'*30)
# agg 대신 함수 직접 작성
def summary_func(gr):
    return{
        'count':gr.count(),  # .count는 numpy함수
        'mean':gr.mean(),
        'std':gr.std(),
        'max':gr.max(),
    }
    
print(group_col.apply(summary_func))  # stack구조로 나옴
print(group_col.apply(summary_func).unstack())

print('\n병합(merge)')
df1 = pd.DataFrame({'data1':range(7),'key':['b','b','a','c','a','a','b']})
print(df1)

df2 = pd.DataFrame({'key':['a','b','d'], 'data2':range(3)})
print(df2)
print()
print(pd.merge(df1, df2))  # key를 기준으로 inner join(교집합처리)
print(pd.merge(df1, df2, on='key', how='inner'))  # 상동
print()
print(pd.merge(df1, df2, on='key', how='outer'))  # full outer join,, outer를 사용함으로써 정수가 실수화가 됨
print()
print
print(pd.merge(df1, df2, on='key', how='left'))   # left outer join
print()
print(pd.merge(df1, df2, on='key', how='right'))   # right outer join

print('공통 칼럼이 없는 경우 : df1 vs df3')
df3 = pd.DataFrame({'key2':['a','b','d'], 'data2':range(3)})
print(df3)
print(pd.merge(df1, df3, left_on='key', right_on='key2'))

print('\nDataFrame 자료 이어 붙이기')
print(pd.concat([df1, df3], axis=0)) # 행 단위 - default
print(pd.concat([df1, df3], axis=1)) # 열 단위

print('\nSeries 병합')
s1 = pd.Series([0,1], index=['a','b'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index=['f','g'])
print(pd.concat([s1,s2,s3], axis=0))

print('그룹 연산 : pivot table - 데이터 열 중에서 두 개의 열(key)을 사용해 데이터를 재구성하여 새로운 집계표를 작성할 수 있다.')
data = {'city':['강남','강북','강남','강북'],
        'year':[2000,2001,2002,2002],
        'pop':[3.3,2.5,3.0,2.0]
}
df = pd.DataFrame(data)
print(df)
print()
print(df.pivot(index='city',columns='year',values='pop')) # pivot(행,열,연산)
# pivot을 사용하니 year, city별로 출력됨
print(df.pivot(index='year',columns='city',values='pop'))
print()
hap = df.groupby(['city'])
print(hap.sum())
print(df.groupby(['city']).sum())  # 위 두줄을 한줄로 표현
print(df.groupby(['city']).agg('sum'))
print(df.groupby(['city']).agg(['sum','mean']))

print(df.groupby(['city','year']).mean())

print('DataFrame.pivot_table : pivot과 groupby 명령의 중간적 성격')
print(df)
print(df.pivot_table(index=['city']))  # 기본 연산은 np.mean이다.
print(df.pivot_table(index=['city'],aggfunc=np.mean))  # 상동

print(df.pivot_table(index=['city','year'],aggfunc=[len,np.sum]))  # city별, year별 개수와 합 구하기

print(df.pivot_table(values=['pop'], index=['city'])) # city별 pop의 평균
print(df.pivot_table(values=['pop'], index=['city'], aggfunc=np.mean))  # 상동

print(df.pivot_table(values=['pop'], index='year', columns='city')) # year별 city별 pop의 평균
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], margins=True)) # margin을 사용해 합을 구함
print(df.pivot_table(values=['pop'], index=['year'], columns=['city'], margins=True, fill_value=0)) # NaN를 0으로 채움