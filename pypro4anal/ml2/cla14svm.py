# 서포트 벡터머신(SVM) 알고리즘
# SVM 알고리즘은 머신러닝 알고리즘에서 가장 유명하고 많이 사용되고 있다.
# SVM 알고리즘은 기본적으로 두 개의 그룹(데이터)을 분리하는 방법을 데이터들과 거리가 가장 먼 초평면을 선택하여 분리하는 방법이다.


# Support vector 확인해보기 

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='malgun gothic')

 
X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.5, random_state=4)
y = 2 * y - 1

 

plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("학습용 데이터")
plt.show()

 

from sklearn.svm import SVC
model = SVC(kernel='linear', C=1.0).fit(X, y)  # tuning parameter  값을 변경해보자.

 

xmin = X[:, 0].min()
xmax = X[:, 0].max()
ymin = X[:, 1].min()
ymax = X[:, 1].max()
xx = np.linspace(xmin, xmax, 10)
yy = np.linspace(ymin, ymax, 10)
X1, X2 = np.meshgrid(xx, yy)

 

z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):    # 배열 좌표와 값 쌍을 생성하는 반복기를 반환
    x1 = val
    x2 = X2[i, j]
    p = model.decision_function([[x1, x2]])
    z[i, j] = p[0]

 

plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', label="-1 클래스")
plt.scatter(X[y == +1, 0], X[y == +1, 1], marker='x', label="+1 클래스")
plt.contour(X1, X2, z, levels=[-1, 0, 1], colors='k', linestyles=['dashed', 'solid', 'dashed'])
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, alpha=0.3)

 

x_new = [10, 2]
plt.scatter(x_new[0], x_new[1], marker='^', s=100)
plt.text(x_new[0] + 0.03, x_new[1] + 0.08, "테스트 데이터")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("SVM 예측 결과")
plt.show()

# Support Vectors 값 출력

print(model.support_vectors_)