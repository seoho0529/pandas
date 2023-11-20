# [SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습
# https://www.kaggle.com/zhaoyingzhu/heartcsv
# https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv
# Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
# 각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
# dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
# 임의의 값을 넣어 분류 결과를 확인하시오.     
# 정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ
#
# feature 칼럼 : 문자 데이터 칼럼은 제외
# label 칼럼 : AHD(중증 심장질환)
#
# 데이터 예)
# "","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
# "1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
# "2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
# ...
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Heart.csv')
print(df.head(3), df.shape)
print(df.info())

print(df['ChestPain'].unique())
print(df['Thal'].unique())
label = df['AHD']
print(label[:2])

feature = df[["Age","Sex","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca"]]
print(feature)
feature.Ca = feature.Ca.fillna(feature.Ca.mean())
print(feature.Ca)



# train / test split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = svm.SVC(C=0.01).fit(x_train, y_train) # C는 작을수록 좋음
print(model)

pred = model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)

ac_score = metrics.accuracy_score(y_test, pred)
print('분류 정확도 : ', ac_score)


from sklearn import model_selection
cross_vali = model_selection.cross_val_score(model, feature, label, cv=3)
print('각각의 검증 정확도 : ', cross_vali)
print('평균 검증 정확도 : ', cross_vali.mean())

# 새로운 값을 입력받아 분류 예측
new_Age = float(input("나이를 입력하세요: "))
new_Sex = float(input("성별을 입력하세요 (남성: 1, 여성: 0): "))
new_RestBP = float(input("안정 휴식 혈압을 입력하세요: "))
new_Chol = float(input("new_Chol 입력하세요: "))
new_Fbs = float(input("Fbs 입력하세요: "))
new_RestECG = float(input("RestECG 입력하세요: "))
new_MaxHR = float(input("MaxHR 입력하세요: "))
new_ExAng = float(input("ExAng 입력하세요: "))
new_Oldpeak = float(input("Oldpeak 입력하세요: "))
new_Slope = float(input("Slope 입력하세요: "))
new_Ca = float(input("Ca 입력하세요: "))

newdata = pd.DataFrame({
    "Age": [new_Age],
    "Sex": [new_Sex],
    "RestBP": [new_RestBP],
    "Chol": [new_Chol],
    "Fbs": [new_Fbs],
    "RestECG": [new_RestECG],
    "MaxHR":[new_MaxHR],
    "ExAng":[new_ExAng],
    "Oldpeak":[new_Oldpeak],
    "Slope":[new_Slope],
    "Ca":[new_Ca]
})

# 새로운 예측결과
newPred = model.predict(newdata)
print('새로운 예측 결과인 AHD는 : ', newPred)

import seaborn as sns
import matplotlib.pyplot as plt

