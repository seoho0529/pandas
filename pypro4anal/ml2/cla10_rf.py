# RandomForestClassifier : DecisionTree 여러 개를 합쳐서 앙상블 모델로 운영 - Bagging 알고리즘
# 당연히 DecisionTree 보다 성능이 우수하다.
# DecisionTree 결과값을 Voting을 통해 얻는다. Boosting에 비해 성능은 다소 떨어지나 과적합 처리가 효과적이다.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# titanic data으로 실습
df = pd.read_csv("../testdata/titanic_data.csv")
print(df.head(3), df.shape) # (891, 12)
print(df.isnull().any())

df = df.dropna(subset=['Pclass', 'Age', 'Sex'])

df_x = df[['Pclass', 'Age', 'Sex']]  # 2차원으로 바꿔줌
print(df_x.head(2), df_x.shape) # (714, 12)

df_y = df['Survived']
print(df_y.head(2))

print()
print(df_x['Sex'][:2])  # male:1, female:0 으로 dummy 변수화
from sklearn.preprocessing import LabelEncoder  # Categorical데이터를 Numerical로 변환, LabelEncoder는 사전순으로 숫자가 나오기 때문에 female이 먼저 0을 받고 male이 1
df_x.loc[:,'Sex'] = LabelEncoder().fit_transform(df_x['Sex'])
# df_x['Sex'] = df_x['Sex'].apply(lambda x:1 if x=='male' else 0)
print(df_x['Sex'][:2])
# pd.set_option('display.max_columns', 500)
print(df_x.head(2))

# train / test split
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.25, random_state=12)

# model - 랜덤포레스트
model = RandomForestClassifier(criterion='entropy', n_estimators=500)
model.fit(train_x, train_y)

pred = model.predict(test_x)
print('예측값 : ', pred[:5])
print('실제값 : ', test_y[:5].ravel())

print('acc : ', sum(test_y == pred) / len(test_y))
from sklearn.metrics import accuracy_score
print('acc : ', accuracy_score(test_y, pred))  # 정확도 :  0.8212290502793296

# KFold 교차검증
from sklearn.model_selection import cross_val_score
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)
print('KFold 교차검증 5회 실시한 정확도 평균 : ', np.round(np.mean(cross_vali),3)) # 0.81
# Bagging(과적합 추정 모델을 결합해 이 과적합 효과를 줄임) - 임의화 의사결정 트리 앙상블 모델 : 예)RandomForestClassifier...
# 임의화(randomization) : 확률적 메커니즘을 통해 실험 개체를 비교하는 두 그룹
# 즉, 처리되는 그룹과 대조되는 그룹에 배치하도록 하는 것을 의미한다. 임의화 샘플링(복원추출) 


# 중요 변수(feature) 확인
print('특성(변수, feature) 중요도 : ', model.feature_importances_) # [0.15284342 0.5477223  0.29943428]
# Age, Pclass, Sex 순

# feature 중요도 시각화
import matplotlib.pyplot as plt
def plot_feature_importance_func(model):
    n_features = df_x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center') # 막대그래프 세로
    plt.yticks(np.arange(n_features), df_x.columns)
    plt.xlabel('importance')
    plt.ylabel('feature name')
    plt.show()

plot_feature_importance_func(model)
