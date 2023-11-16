# Decision Tree : classification, regression 둘 다 가능하지만 주로 분류에서 사용된다.
# 데이터 균일도에 따른 규칙기반의 결정트리
# 간단한 알고리즘에 성능이 우수하다. 데이터의 양이 많아질수록 성능은 떨어진다.
# 전체 자료를 계속 조건에 의해 양분하는 분류기법. 불순물이 없을 때 까지 분류를 계속 진행

# 시각화를 위해 graphviz와 pydot 패키지가 필요

import pydotplus
from sklearn import tree

# 키, 머리카락 길이로 남여 구분
x = [[180,15],[177,20],[156,35],[174,5],[166,33],[190,5],[167,12],[159,25],[184,10],[166,5]]
y = ['men', 'women', 'women', 'men', 'women', 'men', 'men', 'men', 'men', 'women']
label_names = ['height','hair_length']

model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0).fit(x,y)
# entropy : 데이터 집합에서 혼잡도(불순도)를 말하며 클수록 불순도가 높다.
print(model)
pred = model.predict(x)
print('예측값 : ',pred)
print('실제값 : ',y)
print('훈련 정확도 : ', model.score(x,y))


# 시각화
import collections
dot_data = tree.export_graphviz(model, feature_names=label_names, out_file=None, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
colors = ('red', 'orange')
edges = collections.defaultdict(list)

for e in graph.get_edge_list():
    edges[e.get_source()].append(int(e.get_destination()))

for e in edges:
    edges[e].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[e][i]))[0]
        dest.set_fillcolor(colors[i])
        
graph.write_png('cla6tree.png')  # 이미지를 png로 저장
        
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
img = imread('cla6tree.png')    
plt.imshow(img)        
plt.show()

new_pred = model.predict([[170, 120]])
print('예측 결과 : ', new_pred)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        