# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

data = {
    '요일': ['토', '토', '토', '화', '토', '월', '토', '토', '토', '토', '토', '토', '토', '일', '월', '화', '수', '목', '금', '토', '토', '일', '월', '화', '수', '목', '금', '토'],
    '외식유무': [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    '소득수준': [57, 39, 28, 60, 31, 42, 54, 65, 45, 37, 98, 60, 41, 52, 75, 45, 46, 39, 70, 44, 74, 65, 46, 39, 60, 44, 30, 34]
}
df = pd.DataFrame(data)
print(df)

train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
print(train.shape, test.shape)

# 분류모델
formula='외식유무~소득수준'
model = smf.logit(formula=formula, data=df).fit()
# print(model)
print(model.summary())

pred = model.predict(df[:10])
print('예측값 : ', pred.values)
print('예측값 : ', np.around(pred.values))
print('실제값 : ', df['외식유무'][:10].values)

conf_tab = model.pred_table()
print('confusion matrix : \n', conf_tab)

print('분류 정확도 : ',(16+10) / len(df))

newdf = pd.DataFrame({'소득수준': [int(input('소득수준 입력: '))]})
flag = np.rint(model.predict(newdf))[0]
print('외식함' if flag == 1 else '외식안함')
'''
so=int(input('소득수준: '))
cla=np.rint(model.predict(so))[0]
print('소득수준:{}일때 외식은:{}'.format(so, np.round(cla)))
'''