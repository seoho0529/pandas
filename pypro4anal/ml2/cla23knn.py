# 주변의 가장 가까운 K개의 데이터를 보고 데이터가 속할 그룹을 판단하는 알고리즘이 K-NN 알고리즘이다.
# feature의 수가 많거나, 이상치가 있으면 성능이 떨어진다.
# 서로 다른 값들의 비율(단위)이 일정하지 않으면 성능이 떨어지므로 스케일링(정규화,표준화)를 권장한다.

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

train = [  # 단위의 차이가 별로 나지 않기 때문에 스케일링을 하지 않았다. 하지만 KNN은 스케일링을 웬만하면 해주는 것을 추천!
    [5,3,2],
    [6,4,2],
    [3,2,1],
    [2,4,6],
    [1,3,7]
]
label=[0,0,0,1,1]

# plt.xlim([-1,5])
# plt.ylim([0,10])
# plt.plot(train,'o')
# plt.show()

kmodel = KNeighborsClassifier(n_neighbors=1)  # 5일떄 과소적합이 의심됨
kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred : ', pred)
print('test acc : {:.2f}'.format(kmodel.score(train, label)))

newData = [[1,2,15], [78,22,-5]]
newpred = kmodel.predict(train)
print('newpred : ', newpred)