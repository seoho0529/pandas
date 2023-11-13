import pandas as pd
import joblib


# 방법2 - 메모리가 더 절약되는 방법2를 더 선호

mymodel = joblib.load('linear6ex_m.model')  
    
# 예측2 : 새로운 Advertising, Price 값으로 Sales를 추정(구간추정)
# 다중회귀식 : Sales = 0.1231 * Advertising + (-0.0546) * Price + 13.0034
x_new2 = pd.DataFrame({'Advertising':[1000, 2000, 3000], 'Price':[100, 200, 300]})
new_pred2 = mymodel.predict(x_new2)
print('sales 추정값 : ', new_pred2.values)