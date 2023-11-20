# SVM : 데이터 분류 및 예측을 위한 가장 큰 폭의 경계선을 찾는 알고리즘을 사용
# '커널트릭' 이라는 기술을 통해 선형은 물론 비선형, 이미지 분류 까지도 처리가 가능하다.

# SVM으로 XOr 처리를 실습

x_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,1]
]

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics

df = pd.DataFrame(x_data)
print(df)

feature = np.array(df.iloc[:, 0:2])
label = np.array(df.iloc[:, 2])  # vector가 됨
print(feature)
print(label)

print()
model1 = LogisticRegression().fit(feature, label)
pred = model1.predict(feature)
print('Logistic 예측값 : ',  pred)
print('Logistic 정확도 : ',  metrics.accuracy_score(label, pred))  # 실제값,예측값


# model2 = svm.SVC().fit(feature, label)  # 알고리즘 이름은 SVM인데 실질적인 클래스 이름은 SVC이다.
model2 = svm.LinearSVC().fit(feature, label) # 선형일땐 svc보단 linearsvc가 성능이 더 좋고 빠르지만, 비선형일떈 svc를 사용한다. 떄문에 선형,비선형이 가능한 svc를 사용하는 편
pred2 = model2.predict(feature)
print('SVM 예측값 : ',  pred2)
print('SVM 정확도 : ',  metrics.accuracy_score(label, pred2))


