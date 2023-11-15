# [로지스틱 분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('../testdata/bodycheck.csv')
# print(df.head(3), df.shape)

# TV시청, 안경유무 컬럼만 추출
df=df[['게임','TV시청','안경유무']]
print(df)

# 독립변수 : 게임, TV시청  |  종속변수 : 안경유무
# 모델 학습을 위해 train/test로 분리하기
x = df[['게임','TV시청']]
y = df['안경유무']
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (14, 2) (6, 2) (14,) (6,)

# 모델작성
model = LogisticRegression(C=1.0, solver='lbfgs', multi_class='auto', random_state=0, verbose=1)
print(model)
model.fit(x_train, y_train)

# 분류예측
print('test로 정확도는 ', model.score(x_test, y_test))         # test로 정확도는  1.0
print('train으로 정확도는 ', model.score(x_train, y_train))     # train으로 정확도는  1.0

# 예측에 사용할 새로운 데이터 생성
newdf = pd.DataFrame({'게임':[int(input('게임시간 입력 : '))], 'TV시청':[int(input('TV시청시간 입력 : '))]})
print("입력한 데이터:")
print(newdf)

# 모델을 사용하여 예측
pred = model.predict(newdf)

# 예측 결과 출력
if pred[0] == 0:
    print("안경 미착용 (값: 0)")
else:
    print("안경 착용 (값: 1)")


