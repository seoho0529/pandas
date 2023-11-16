# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 
# 참여 칼럼 : 
#   Daily Time Spent on Site : 사이트 이용 시간 (분)
#   Age : 나이,
#   Area Income : 지역 소독,
#   Daily Internet Usage:일별 인터넷 사용량(분),
#   Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간의 단위가 큰 경우 표준화 작업을 시도한다.
# ROC 커브와 AUC 출력

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import numpy as np
import pandas as pd

df = pd.read_csv('../testdata/advertisement.csv')
print(df.head(3), df.shape)  # (1000, 10)

# 필요한 칼럼 추출
df=df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Clicked on Ad']]
print(df.head(3), df.shape)  # (1000, 5)


# 독립변수, 종속변수 | train, test 데이터로 분리하기
x = df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage']]
y = df['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (700, 4) (300, 4) (700,) (300,)

# 표준화
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])


# 모델 작성
model = LogisticRegression(C=1.0, solver='lbfgs', multi_class='auto', random_state=0, verbose=0)
model.fit(x_train, y_train)

# 분류예측
y_pred = model.predict(x_test)  # 검증은 test로
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 갯수 : %d, 오류 수 : %d'%(len(y_test), (y_test != y_pred).sum()))  # 총 갯수 : 45, 오류 수 : 4
print('분류 정확도 확인 1')
print('%.5f'%accuracy_score(y_test, y_pred))

# confusion matrix의 값들 보기
from sklearn import metrics 
cl_rep = metrics.classification_report(y_test, y_pred) 
print(cl_rep)


# ROC곡선
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0,1],[0,1], 'k--', label='random classifier line(AUC:0.5)')
plt.xlabel('fpr', fontdict={'fontsize':14})
plt.ylabel('tpr')
plt.legend() # label을 주었으니
plt.show()

# AUC
print('AUC : ', metrics.auc(fpr, tpr))  #AUC :  0.7728079999999999



