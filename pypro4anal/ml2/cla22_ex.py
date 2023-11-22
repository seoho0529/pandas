'''
[GaussanNB 문제]
독버섯(poisonous)인지 식용버섯(edible)인지 분류
https://www.kaggle.com/datasets/uciml/mushroom-classification
feature는 중요변수를 찾아 선택, label:class
참고 : from xgboost import plot_importance

데이터 변수 설명 : 총 23개 변수가 사용됨.
여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).
변수명 변수 설명
class      edible = e, poisonous = p
cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
bruises        bruises = t, no = f
odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
gill-attachment attached = a, descending = d, free = f, notched = n
gill-spacing close = c, crowded = w, distant = d
gill-size       broad = b, narrow = n
gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
stalk-shape  enlarging = e, tapering = t
stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
veil-type      partial = p, universal = u
veil-color     brown = n, orange = o, white = w, yellow = y
ring-number none = n, one = o, two = t
ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d
'''
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../testdata/mushrooms.csv')

# class를 제외한 독립변수 선택
features = df.drop('class', axis=1)
features = pd.get_dummies(features)
label = df['class'].map({'p':1, 'e':0})


# 학습 및 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=12)

# XGBClassifier 모델 생성 및 학습
model = XGBClassifier()
model.fit(X_train, y_train)

# 특성 중요도 시각화
plot_importance(model)
plt.show()
print(df.columns)

features2 = features[['gill-size_b', 'odor_n', 'bruises_f']].copy()


# 학습 및 테스트 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(features2, label, test_size=0.2, random_state=6413)

# GaussianNB 모델 생성 및 학습
gmodel = GaussianNB()
gmodel.fit(train_x, train_y)

# 예측
pred = gmodel.predict(test_x)
print('예측값 : ', pred[:10])
print('실제값 : ', test_y[:10])

# 정확도 평가
acc = sum(test_y == pred) / len(pred)
print('정확도 : ', acc) # 0.9415
print('정확도 : ', accuracy_score(test_y, pred)) # 0.9415
print('분류 보고서 : \n', classification_report(test_y, pred))

print('새 값으로 예측')
new = np.array([[2, 12, 0],[22, 34, 50], [1,2,3]])
print('예측 결과 : ', gmodel.predict(new))