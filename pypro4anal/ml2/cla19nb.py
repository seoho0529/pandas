# Naive Bayes Classifier란?
# 나이브 베이즈 분류기는 각 사건 특성들이 독립이라는 가정 하에 베이즈 정리(Bayes’ theorem)를 적용한 간단한 확률 기반 분류, 예측이다.
# 이는 일반적으로 결과의 확률을 추정하기 위해 많은 속성을 고려해야 하는 문제에 가장 적합하다.

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn import metrics

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]  # newaxis : 차원확장
print(x)
y = np.array([1,3,5,7,9])
print(y)

model = GaussianNB().fit(x,y)
print(model)

pred = model.predict(x)
print(pred)
print('acc : ', metrics.accuracy_score(y, pred))

# 새로운 값으로 분류
newX = np.array([[0.5],[2],[8],[9],[12],[0.1]])
newPred = model.predict(newX)
print(newPred)