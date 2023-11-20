'''
kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)
dataset은 winequality-red.csv 
https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv
 
Input variables (based on physicochemical tests):
 1 - fixed acidity
 2 - volatile acidity
 3 - citric acid
 4 - residual sugar
 5 - chlorides
 6 - free sulfur dioxide
 7 - total sulfur dioxide
 8 - density
 9 - pH
 10 - sulphates
 11 - alcohol
 Output variable (based on sensory data):
 12 - quality (score between 0 and 10)
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn import metrics

data = pd.read_csv('../testdata/winequality-red.csv')
print(data.head(3), data.shape)  # (1596, 12)

x = data.drop('quality', axis=1)
y = data['quality']

#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# 모델 생성
model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=12)
model.fit(x_train, y_train)

# 교차검증 (KFold)
from sklearn.model_selection import cross_val_score
cross_vali = cross_val_score(model, x, y, cv=5)
print(cross_vali)
print('교차검증 5회 실시한 정확도 평균 : ', np.mean(cross_vali))

y_pred = model.predict(x_test)
print('예측값 : ', y_pred[:5])
print('실제값 : ', y_test[:5].ravel())

# 분류 보고서 출력
acc = metrics.accuracy_score(y_test, y_pred)
print('총 개수 : %d, 오류수 : %d'%(len(y_test),(y_test != y_pred).sum()))  # 총 개수 : 320, 오류수 : 92
print('acc :', acc)   # acc : 0.7125
print(metrics.classification_report(y_test, y_pred, zero_division=0))


# 중요 변수(feature) 확인
print('특성(변수, feature) 중요도 : ', model.feature_importances_)

# feature 중요도 시각화
def plot_feature_importance_func(model):
    n_features = x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel('importance')
    plt.ylabel('feature name')
    plt.show()

plot_feature_importance_func(model)


