# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'          testdata 폴더 : glass.csv
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# RI: 굴절률 Na: 나트륨 Mg: 마그네슘 알루미늄: 알루미늄 Si: 실리콘 K: 칼륨 Ca: 칼슘 Ba: 바륨 Fe: 철    
#  Type  유리 유형: (클래스 속성)
# -- 1 Building_windows_float_processed
# -- 2 Building_windows_non_float_processed
# -- 3 vehicle_windows_float_processed
# -- 4 vehicle_windows_non_float_processed (이 데이터베이스에는 없음)
# -- 5 containers
# -- 6 tableware
# -- 7 headlamps
#                           ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost
from xgboost import plot_importance


df = pd.read_csv('../testdata/glass.csv')
print(df.head(3), df.shape) # (214, 10)

x = df.drop(columns = ['Type'])
y = df['Type']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # Type 열은 문자열로 된 클래스 레이블을 포함. 이를 숫자로 변환하기 위해 LabelEncoder를 사용.
y = le.fit_transform(y) # fit_transform을 사용하여 클래스 레이블을 숫자로 변환.

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (149, 9) (65, 9) (149,) (65,)


# model
model = xgboost.XGBClassifier(booster = 'gbtree', max_depth = 6, n_estimators = 500).fit(x_train, y_train) # 의사결정 기반(booster)
pred = model.predict(x_test)
print('예측값 :', pred[:10])
print('실제값 :', y_test[:10])

print('정확도 확인 방법 1')
from sklearn import metrics
acc = metrics.accuracy_score(y_test, pred)
print('acc :', acc)

# 중요 변수 시각화
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(model, ax = ax)
plt.show()
