# 인공신경망(Artifical Neural Network, ANN)은 사람의 뇌 속 뉴런의 작용을 본떠 패턴을 구상한 컴퓨팅 시스템의 일종이다.
# 퍼셉트론은 가장 단순한 유형의 인공 신경망이다. 이런 유형의 네트워크는 대개 이진법 예측을 하는 데 쓰인다.
# 퍼셉트론(단층신경망, 뉴런(노드))은 데이터를 선형적(wx+b)으로 분리할 수 있는 경우에만 효과가 있다.
# w가 최소가 되는 지점을 찾기 위해 비용 함수(cost function)를 사용한다. -> 예측값과 실제값의 차이가 줄어들어 정확한 예측이 가능

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

feature = np.array([[0,0],[0,1],[1,0],[1,1]])
print(feature)
# label = np.array([0,0,0,1])  # and
# label = np.array([0,1,1,1])  # or
label = np.array([0,1,1,0])  # xor

ml = Perceptron(max_iter=10, eta0=0.1, random_state=0).fit(feature, label) 
# 학습반복횟수 : max_iter (딥러닝에선 epoch라고 부름)| eta0(learning rate) : 학습량
print(ml)
pred = ml.predict(feature)
print('pred : ', pred)
print('acc : ', accuracy_score(label, pred))

print('\n다층 신경망 : MLP')
from sklearn.neural_network import MLPClassifier
ml2 = MLPClassifier(hidden_layer_sizes=(30), max_iter=10, solver='adam', learning_rate_init=0.1, verbose=1)
ml2.fit(feature,label)
print(ml2)  # 학습할때마다 loss값이 점점 떨어지며 최소가 되는 w를 찾아가는중
pred2 = ml2.predict(feature)
print('pred2: ', pred2)
print('acc2 : ', accuracy_score(label, pred2))