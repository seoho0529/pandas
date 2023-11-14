from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('../testdata/Consumo_cerveja.csv') # usecols=[]를 사용하여 필요한 칼럼만 처음부터 보기
print(df.head(3), df.shape)
print(df.isnull())

# NaN이 있는 경우 삭제
df = df.dropna(axis=0)
print(df, df.shape)

# ,을 .로 바꾸기
for column in ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)']:
    df[column] = df[column].astype(str).str.replace(',', '.')
print(df['Temperatura Media (C)']) # 바뀌었는지 확인

# feature : Temperatura Media (C) : 평균 기온(C)
#           Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오

# dataset 분리
train, test = train_test_split(df, test_size=0.4, random_state=12)
print(train)
print(test)

x_train = train[['Temperatura Media (C)','Precipitacao (mm)']]
y_train = train['Consumo de cerveja (litros)']

x_test = test[['Temperatura Media (C)','Precipitacao (mm)']]
y_test = test['Consumo de cerveja (litros)']

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (219, 2) (146, 2) (219,) (146,)

# LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('실제값 : ', y_test.values)
print('예측값 : ', np.round(y_pred, 0))

# 결정계수 수식 사용
# 잔차 구하기
y_mean = np.mean(y_test)
nomerator = np.sum(np.square(y_test - y_pred))
denomerator = np.sum(np.square(y_test - y_mean))

r2 = 1 - nomerator / denomerator
print('결정계수 : ', r2)
print('결정계수 : ', r2_score(y_test, y_pred))
# 결정계수 :  0.393이므로 약 39.3%의 모형 설명력을 가지고 있다.

feature = df[['Temperatura Media (C)','Precipitacao (mm)']]
print(feature[:10])
label = df[['Consumo de cerveja (litros)']]
print(label[:10])

lmodel = LinearRegression().fit(feature, label)
print('회귀계수_기울기(slope)', lmodel.coef_) # [[ 0.80190566 -0.07366404]]     
print('회귀계수_절편(intercept)', lmodel.intercept_) # [8.76264285]
# 다중회귀식 : Consumo de cerveja (litros) = (0.80190566)*Temperatura Media (C) + (-0.07366404)*Precipitacao (mm) + 8.76264285

pred = lmodel.predict(feature)
print('예측값 : ', np.round(pred[:5],1))
print('실제값 : ', label[:5])

# 모델 성능 평가
print('MSE : ', mean_squared_error(label, pred))
print('결정계수 r2_score : ', r2_score(label, pred))

# 맥주소비량(리터) 예측하기
new_Temperatura = float(input("평균 기온을 입력하세요: "))
new_Precipitacao = float(input("강수량을 입력하세요: "))
new_pred = lmodel.predict([[new_Temperatura, new_Precipitacao]])
print('평균기온:{}도 이며 강수량:{}mm 일때 맥주 소비량은 :{}이다.'.format(new_Temperatura, new_Precipitacao, new_pred[0]))
