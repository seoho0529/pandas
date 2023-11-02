# matplotlib 스타일 인터페이스 | 차트 종류 몇 가지 경험하기

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False

"""
x = np.arange(10)
# matplotlib 스타일 인터페이스 
plt.figure()  # plot을 생성하라는 뜻
plt.subplot(2,1,1)  # 2행 1열의 1행  (row, column, paner number)  영역을 paner number이라 칭함
plt.plot(x, np.sin(x))
plt.subplot(2,1,2)
plt.plot(x, np.cos(x))
plt.show()

# matplotlib 객체 지향 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)  # fig는 plot 전체, ax는 plot 한개 한개를 의미
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()
"""

data = [50, 80, 100, 70, 88]

"""
# 막대 그래프 - 데이터의 양이 많을땐 막대그래프 사용을 잘 하지 않음,, 서열척도, 등간척도 데이터일때 주로 사용
plt.bar(range(len(data)), data)
plt.show()

# 세로 막대 그래프
err = np.random.rand(len(data))
plt.barh(range(len(data)), data, xerr=err)  # error bar(오차막대) : 표준편차, 오차, 신뢰구간 등 표현에 효과적 
plt.show()

# 원 그래프
plt.pie(data, explode=(0,0,0.2,0,0), colors=['yellow','blue','red'])
plt.show()
"""

"""
# 많은 양의 데이터일땐 히스토그램이나 boxplot을 주로 사용

# 히스토그램은 연속형 데이터일때 주로 사용 
plt.hist(data, bins=5, alpha=0.1)  # bins : n개의 구간으로 나누기  alpha : 투명도
plt.show()

# boxplot
plt.boxplot(data, notch=True)
plt.show()

# 시계열 데이터 출력
import pandas as pd
fdata = pd.DataFrame(np.random.rand(1000, 4),
                     index=pd.date_range('1/1/2020', periods=1000), columns=list('abcd'))  
# 정규분포를 따르는 데이터 1000개,, 칼럼명을 a, b, c, d
fdata = fdata.cumsum()  #  누적합 : cumsum()
print(fdata.head(3))
print(fdata.tail(3))
plt.plot(fdata)
plt.show()

# pandas의 plot 기능
fdata.plot(kind='box')
plt.show()
"""

# matplotlib의 기능 보충용 라이브러리로 seaborn
import seaborn as sns
"""
# Seaborn 데이터셋 목록
# print(sns.get_dataset_names())

titanic = sns.load_dataset('titanic')
# print(titanic.info())
print(titanic.head(3))

plt.hist(titanic['age'])
plt.show()

# 위의 matplotlib 방법 말고 seaborn 방법
sns.displot(titanic['age'])
plt.show()

sns.boxplot(y='age',data=titanic)
plt.show()
"""
# iris dataset
iris_data = sns.load_dataset('iris')
print(iris_data.head(3))

import pandas as pd
iris_data = pd.read_csv('../testdata/iris.csv')
print(iris_data.head(3))

# 산점도
plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.show()

# Species별로 색상 부여
print(iris_data['Species'].unique())  # 중복을 확인할땐 unique or set 사용 : ['setosa' 'versicolor' 'virginica']
print(set(iris_data['Species']))    # {'setosa', 'versicolor', 'virginica'}

cols=[]
for s in iris_data['Species']:
    choice = 0
    if  s == 'setosa':choice=1
    elif s == 'versicolor':choice=2
    elif s == 'virginica':choice=3
    cols.append(choice)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.show()


# Sepal.Length  Sepal.Width  Petal.Length  Petal.Width를 사용해 scatter matrix 그래프 출력
iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Width']  # iloc는 인덱스, loc는 칼럼
print(iris_col)

pd.plotting.scatter_matrix(iris_col, diagonal='kde') # 밀도곡선 보기 위해 diagonal을 사용
plt.show()

# seaborn으로 scatter matrix 그래프 출력
sns.pairplot(iris_data, hue='Species', height=1)  # 카데고리 칼럼(hue)
plt.title('seaborn으로 scatter matrix 그래프 출력')
# plt.show()

# 그래프(차트)를 이미지로 저장
fig = plt.gcf()  # 저장용 객체 fig
plt.show()
fig.savefig('plot2.png')

# 이미지 읽기
from matplotlib.pyplot import imread
img = imread('plot2.png')
plt.imshow(img)
plt.show()