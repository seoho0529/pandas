# Ensemble 학습 : 개별적으로 동작하는 여러 모델들을 종합하여 예측한 결과를 투표에 의해 가장 좋은 결과로 취하는 방법
# LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier 분류기를 합쳐 예측한 결과를 투표로 결정

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier   # 최근접 이웃 알고리즘
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

cancerData = load_breast_cancer()
dataDf = pd.DataFrame(cancerData.data, columns=cancerData.feature_names)
print(dataDf.head(3), dataDf.shape)
print(set(cancerData.target))  # {0, 1}
print(cancerData.target_names) # ['malignant' 'benign'] 음성:0, 양성:1


x_train, x_test, y_train, y_test = train_test_split(cancerData.data, cancerData.target, test_size=0.2, random_state=911)

logiModel = LogisticRegression()
knnModel = KNeighborsClassifier(n_neighbors = 3)
decModel = DecisionTreeClassifier()


# 앙상블 기법 사용 - voting유형 - 튜플 구조로 3개의 모델을 입력해줌, soft방식
classifires = [logiModel, knnModel, decModel]

for cl in classifires:  # 각각 돌림
    cl.fit(x_train, y_train)  #  각자 학습해줘야함
    pred = cl.predict(x_test) # test로 검정
    class_name = cl.__class__.__name__ # 하면 LR, KNN, Decision이 나올것
    print('{0} 정확도:{1:.4f}'.format(class_name, accuracy_score(y_test, pred)))

votingModel = VotingClassifier(estimators=[('LR',logiModel),('KNN',knnModel),('Decision',decModel)], voting='soft')
votingModel.fit(x_train, y_train)

vpred = votingModel.predict(x_test)
print('Voting 분류기의 정확도 : {0:.4f}'.format(accuracy_score(y_test, vpred)))

