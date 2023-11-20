# 주성분 분석(Principal Component Analysis, PCA)은 가장 널리 사용되는 차원 변형 기법(특성공학:feature engineering) 중 하나로, 
# 원 데이터의 분포를 최대한 보존하면서 고차원 공간의 데이터들을 저차원 공간으로 변환한다.
# PCA는 기존의 변수를 조합(국어 + 영어 -> 어학)하여 서로 연관성이 없는 새로운 변수, 즉 주성분(principal component, PC)들을 만들어 낸다.
# n개의 관측치와 p개의 변수로 구성된 데이터를 상관관계가 없는 k개의 변수로 구성된 데이터로 요약하는 방식.
# 이때 요약된 변수는 기존 변수의 선형조합으로 원래 성질을 최대한 유지해야 한다.
# 선현대수 관점으로 볼 때 입력데이터의 공분산 행렬을 고윳값 분해하고 이렇게 구한 고유벡터에 입력데이터를 선형변환하는 것 이다.
# 이 고유벡터(행렬을 곱하더라도 방향이 변하지 않고 크기만 변하는 벡터)가 PCA의 주성분벡터로서 입력데이터의 분산이 큰 방향을 나타낸다.

# iris dataset을 사용
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
plt.rc('font', family='malgun gothic')


iris = load_iris()
print(iris.feature_names)
n = 10 # 관측치 10개
x = iris.data[:n, :2] # sepal - length, width
# print('차원축소 전 x :',x,x.shape,x,type(x))
# print(x.T)


# 시각화
# plt.plot(x.T, 'o:')
# plt.grid()
# plt.title('iris 크기 특성')
# plt.xlabel('특성 종류')
# plt.ylabel('추천서')
# plt.legend(['표본{}'.format(i+1) for i in range(n)])
# plt.show()

# scatter plot
df = pd.DataFrame(x)
print(df.head(2))
# ax = sns.scatterplot(x=df[0], y=df[1], data=pd.DataFrame(x), marker='s', s=100, color=['g'])
# for i in range(n):
#     ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i+1))
# plt.xlabel('꽃받침 길이')
# plt.ylabel('꽃받침 너비')
# plt.axis('equal')  # 가로세로 길이 맞추기
# plt.show()

print()
print(x.shape) # (10, 2)
# PCA로 차원축소(꽃받침 길이, 꽃받침 너비 -> 한 개의 값)
pca1 = PCA(n_components=1) # 2개를 한개로
x_low = pca1.fit_transform(x)  # 비지도 학습이므로 y값(target)은 주지 않는다. 
print('x_low : ', x_low[:3], ' ', x_low.shape) # (10, 1)

x2 = pca1.inverse_transform(x_low) # 주성분 분석된 값을 원복 -> 똑같은 값이 아닌 근사한 값으로 됨
print('차원 복귀 후 x2 : ', x2, ' ', x2.shape)

# scatter plot
"""
df = pd.DataFrame(x)
print(df.head(2))
ax = sns.scatterplot(x=df[0], y=df[1], data=pd.DataFrame(x), marker='s', s=100, color=['r'])
for i in range(n):
    d = 0.03 if x[i, 1] > x2[i, 1] else -0.04 # text위치 조절하기 위해
    ax.text(x[i, 0] - 0.06, x[i, 1] + d, '표본{}'.format(i + 1))
    plt.plot([x[i, 0], x2[i, 0]], [x[i, 1], x2[i, 1]], 'k--')
plt.plot(x2[:,0], x2[:, 1], 'o-', color='b', markersize=10)
plt.plot(x[:,0].mean(), x[:, 1].mean(), 'D', color='y', markersize=10)
plt.axvline(x[:, 0].mean(), c='gray')
plt.axhline(x[:, 1].mean(), c='gray')
plt.xlabel('꽃받침 길이')
plt.ylabel('꽃받침 너비')
plt.axis('equal')  # 가로세로 길이 맞추기
plt.show()
"""

print('4개의 열을 2개로 축소')
x = iris.data
pca2 = PCA(n_components=2)  # 2개로 만들려고 하니 2
print('원래 x : ', x[:3], ' ', x.shape)
x_low2 = pca2.fit_transform(x)  # 비지도 학습이므로 y값(target)은 주지 않는다. 
print('x_low2 : ', x_low2[:3], ' ', x_low2.shape) # (150, 4)

x4 = pca2.inverse_transform(x_low2) # 주성분 분석된 값을 원복 -> 똑같은 값이 아닌 근사한 값으로 됨
print('차원 복귀 후 x4 : ', x4[:3], ' ', x4.shape)   # (150, 2),, 근사행렬로 변환
# 원본 : [[5.1 3.5 1.4 0.2], [4.9 3.  1.4 0.2], [4.7 3.2 1.3 0.2]]
# pca : [[-2.68412563  0.31939725], [-2.71414169 -0.17700123], [-2.88899057 -0.14494943]]
# 원복 : [[5.08303897 3.51741393 1.40321372 0.21353169], [4.7462619  3.15749994 1.46356177 0.24024592], [4.70411871 3.1956816  1.30821697 0.17518015]]
print(pca2.explained_variance_ratio_)# [0.92461872 0.05306648], 변동성 비율

print()
'sepal length', 'sepal width', 'petal length', 'petal width'
iris1 = pd.DataFrame(x,  columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
print(iris1.head(3))
iris2 = pd.DataFrame(x_low2,  columns=['var1', 'var2'])
# var1, var2가 sepal끼리, petal끼리 묶은것이 아니기 때문에 가명으로 var1, var2를 써줌
print(iris2.head(3))

print('---------')
from sklearn import svm, metrics
feature1 = iris1.values
print(feature1[:3])
label = iris.target
print(label[:3])

print('원본데이터로 처리')
model1 = svm.SVC(C=0.1, random_state=0).fit(feature1, label)
pred1 = model1.predict(feature1)
print('model1 acc : ', metrics.accuracy_score(label, pred1))


      
feature2 = iris2.values
print(feature2[:3])
label = iris.target
print(label[:3])

print('PCA 데이터로 처리')
model2 = svm.SVC(C=0.1, random_state=0).fit(feature2, label)
pred2 = model2.predict(feature2)
print('model2 acc : ', metrics.accuracy_score(label, pred2)) 
      
# model1 acc :  0.94 | model2 acc :  0.9466 이므로 둘의 차이가 통계적으로 큰 차이가 없기 때문에 둘 중 한 모델을 사용하여도 무방하다.
      
      
# var1, var2
# 주성분 분석(PCA)에서 각 주성분은 원래 데이터의 분산을 가장 크게 하는 방향으로 정의되는데, 
# 이는 데이터의 변동성(variability)을 가장 잘 설명하는 방향으로 설정된다.      
      
      
      
