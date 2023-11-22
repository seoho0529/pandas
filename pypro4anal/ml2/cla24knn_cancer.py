from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

cancer=load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66, stratify=cancer.target)
# stratify = 속성을 사용하면 편향을 방지한다. 분류의 경우에는 성능을 보장받기 위해 사용을 권장

train_acc = []
test_acc = []

k_setting = range(1,11) # 최적의 k개를 얻고자 후보 k를 결정
for n in k_setting:         
    clf = KNeighborsClassifier(n_neighbors=n, p=2, metric='minkowski')
    clf.fit(x_train, y_train)
    train_acc.append(clf.score(x_train, y_train))   # train 정확도
    test_acc.append(clf.score(x_test, y_test))     # test 정확도
    
import numpy as np
print('train 분류 평균 정확도 : ', np.mean(train_acc))
print('test 분류 평균 정확도 : ', np.mean(test_acc))

# 최적의 k를 위한 시각화
plt.plot(k_setting, train_acc, label='훈련데이터 정확도')
plt.plot(k_setting, test_acc, label='검증데이터 정확도')
plt.xlabel('k값')
plt.ylabel('분류 정확도')
plt.legend()
plt.show()