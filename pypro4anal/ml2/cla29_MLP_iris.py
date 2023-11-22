# MLPClassifier 클래스 사용 - iris dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')
plt.rcParams['axes.unicode_minus']=False
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
# print(iris.DESCR)
print(iris.keys())
print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']
print(iris.data[:2])  # matrix
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))  # 0.9628


x = iris.data[:, [2, 3]]  # feature : 'petal length (cm)', 'petal width (cm)'
y = iris.target  # target은 벡터
print(x[:2])
print(y[:2], set(y))  # {0, 1, 2}

print()
# train / test split (7:3)  과적합 or 과소적합 방지 기능
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7, shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

print()
'''
# scaling(크기를 고르게) - feature에 대해 표준화, 정규화 : 최적화 과정에서 안정성, 수렴 속도를 향상, 과적합 or 과소적합 방지 기능
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train)
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])

# 스케일링 값 원복  (똑같지 않을 수 있음 - 근사치로 감)
# 정규화/ 표준화는 feature에 하는거임. label에 하는거 아님(1, 0뿐인데)
# predict할 때 정규화 했으면 정규화 한 값으로 바꿔서 예측치 돌려야됨
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
'''

# 모델 작성 --------------------
model = MLPClassifier(hidden_layer_sizes=(30), max_iter=100, solver='adam', learning_rate_init=0.1, verbose=1)
model.fit(x_train,y_train)  # 학습은 train으로
# ----------------------------


# 분류예측
y_pred = model.predict(x_test)  # 검증은 test로
print('예측값 : ', y_pred)
print('실제값 : ', y_test)
print('총 갯수 : %d, 오류 수 : %d'%(len(y_test), (y_test != y_pred).sum()))  # 총 갯수 : 45, 오류 수 : 4
print('분류 정확도 확인 1')
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)

# print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))


print('분류 정확도 확인 3')
print('test로 정확도는 ', model.score(x_test, y_test))      # 0.911
print('train으로 정확도는 ', model.score(x_train, y_train))  # 0.990


# 모델 성능이 만족스러운 경우 모델 저장 후 
import joblib  # import pickle도 가능
joblib.dump(model, 'cla4model.sav')


del model  # 모델 지우기

mymodel = joblib.load("cla4model.sav")
print("새로운 값으로 분류 예측 - 'petal length (cm)', 'petal width (cm)' - 스케일링해서 학습했다면 예측 데이터도 스케일링 함")
print(x_test[:2])
new_data = np.array([[5.1, 2.4], [0.1, 0.1], [5.6, 5.6], [8.1, 0.5]]) # 스케일링 안헀기 때문에 그냥 값을 넣지만 스케일링을 했다면 transform() 사용
new_pred = mymodel.predict(new_data)  # softmax함수가 반환한 결과에 대해 가장 큰 인덱스를 반환
print('예측 결과 : ', new_pred)
# print('soft 결과(날것) : ', mymodel.predict_proba(new_data))


# 시각화
from matplotlib.colors import ListedColormap
from matplotlib import font_manager, rc
  
plt.rcParams['axes.unicode_minus']= False
def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')        # 점 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)       # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_region(X=x_combined_std, y=y_combined, classifier=mymodel, test_idx=range(105, 150), title='scikit-learn제공') 




