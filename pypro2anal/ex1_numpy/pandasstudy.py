import MySQLdb
import pickle
import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# 1번
import numpy as np
data = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])
print(data[:, ::-1][::-1])

# 2번
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.pivot_table(values='survived', index=['sex'], columns=['pclass']))


# 3번
data = {
    'product':['아메리카노','카페라떼','카페모카'],
    'maker':['스벅','이디아','엔젤리너스'],
    'price':[5000,5500,6000]
}
df = pd.DataFrame(data)
print(df)
# df.to_sql('test',conn, if_exits='append', index=False)

# 4번 
data1=pd.DataFrame(np.arange(12).reshape(4,3), index=['1월','2월','3월','4월'], columns=['강남','강북','서초'])
print(data1)

# 5번
# plt.show()

# 6번
# data = pd.DataFrame(data)
# data.to_csv('test.csv', index=False, header=False)

# 7번
from pandas import DataFrame
frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']},
                  index=['a','b', 'c','d'])
print(frame.T)
frame2 = frame.drop('d', axis=0)
print(frame2)

# 8번
# df=pd.read_csv('ex.csv', names=['a','b','c','d'])

# 9번
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
df = DataFrame(data)
print(df)
results=Series([x.split()[0] for x in df.juso])
print(results)

# 10번
x=np.array([1,2,3,4,5])
y=np.array([[1],[2],[3]])
print(x+y)

# 11번
# for atag in convert_data.findall('a')

# 12번
df = DataFrame([[1.4, np.nan], [7, 4.5], [np.NaN, np.NAN], [0.5, -1]])
print(df.dropna())

# 13번
data = {"a": [80, 90, 70, 30], "b": [90, 70, 60, 40], "c": [90, 60, 80, 70]}
data=pd.DataFrame(data)
print(data)
data.columns = ['국어','영어','수학']
print(data['수학'])
print(data['수학'].std())
print(data[['국어','영어']])

# 14번

# 15번
import scipy.stats as stats
# H0 : 등급에 따라 생존율에 차이가 없다.
# H1 : 등급에 따라 생존율에 차이가 있다.
df=pd.read_csv('../testdata/titanic_data.csv')
print(df)
print(df.isnull().sum())
print(df['Pclass'].unique())
print(df['Survived'].unique())
ctab=pd.crosstab(index=df['Pclass'], columns=df['Survived'])
ctab.index=['앞','중간','끝']
ctab.columns=['살았다','죽었다']
print(ctab)
chi2, p, dof, _ = stats.chi2_contingency(ctab)
print('chi2:{}, p.{}, dof.{}'.format(chi2, p, dof))  # chi2:102.88898875696056, p.4.549251711298793e-23, dof.2
print('p-value 값이 0.05보다 작으므로 대립가설을 채택한다. 즉, 등급에 따라 생존률에 차이가 있다고 할 수 있다.')
