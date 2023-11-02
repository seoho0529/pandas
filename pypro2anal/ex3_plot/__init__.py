# 시각화
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False
"""
# x = ['서울','수원','인천']  # 순서 o, 수정 가능
x = ('서울','수원','인천')  # 순서 o , 수정 불가
# x = {'서울','수원','인천'}  # set은 순서가 없어서 안됨 --> 인덱싱이 안됨
y = [5, 3, 7]
plt.xlim([-1, 3])  # x축 최소값이 -1, 최대값이 3 
plt.ylim([-5, 10])
plt.plot(x,y)
plt.yticks(list(range(-3,11,3)))  # -3부터 10까지 증가치는 3
plt.xlabel('지역명')
plt.ylabel('인구수')
plt.title('선그래프')
plt.show()
"""

"""
# sin곡선
x = np.arange(10)
y = np.sin(x)
print(x,y)
# plt.plot(x,y,'bo')
# plt.plot(x,y,'r+')
plt.plot(x,y,'go--', linewidth=3, markersize=10)
plt.show()
"""

# hold : 복수의 차트 그리기 명령을 하나의 figure에 표현 할 수 있다.
x = np.arange(0, np.pi * 3, 0.1)
print(x)

















