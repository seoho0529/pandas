# 인디언들의 당뇨병 관련 데이터를 이용해 이항분류  :Logistic Regression 클래스 사용

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # sigmoid 함수가 아니라 softmax 함수를 사용해 다항분류가 가능하다.
from sklearn.metrics import accuracy_score

names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
df = pd.read_csv('../testdata/pima-indians-diabetes.data.csv', header=None, names=names)

print(df.head(3))
array = df.values
print(array[:3], array.shape)  # (768, 9)
x = array[:, 0:8]
y = array[:, 8]
print(x[:2], x.shape)  # (768, 8)  matrix
print(y[:2], y.shape)  # (768,)    vector

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (537, 8) (231, 8) (537,) (231,)

"""
# 로지스틱회귀분석 모델 생성 및 학습
model = LogisticRegression().fit(x_train, y_train)
print('예측값 : ', model.predict(x_test[:10]))
print('실제값 : ', y_test[:10])
print((model.predict(x_test) != y_test).sum())  # 실제값과 예측값들의 차이 : 58 ->예측을 58개 틀렸다.

print('test로 검정한 분류 정확도 : ', model.score(x_test, y_test))     # test로 정확도 확인                  0.7489
print('train로 검정한 분류 정확도 : ', model.score(x_train, y_train))  # 학습에 참여한 train 데이터로 정확도 확인 0.7839
# 이때 test,train에 대한 정확도가 비슷하게 나와야 좋음, 
pred = model.predict(x_test)
print('분류 정확도 : ', accuracy_score(y_test, pred))
"""

# 모델 저장
import joblib
# joblib.dump(model, 'cla3model.sav')


# 학습이 끝난 모델 파일 로딩 후 사용
mymodel = joblib.load("cla3model.sav")
print(x_test[:1])
print('분류 예측 : ', mymodel.predict(x_test[:1]))  # 분류 예측 :  [0.]



