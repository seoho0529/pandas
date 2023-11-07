# 서로 대응인 두 집단의 평균 차이 검정(paired samples t-test)
# 처리 이전과 처리 이후를 각각의 모집단으로 판단하여, 동일한 관찰 대상으로부터 처리 이전과 처리 이후를 1:1로 대응시킨 두 집단으로 부터
# 의 표본을 대응표본(paired sample)이라고 한다.
# 대응인 두 집단의 평균 비교는 동일한 관찰 대상으로부터 처리 이전의 관찰과 이후의 관찰을 비교하여 영향을 미친 정도를 밝히는데 주로 사용
# 하고 있다. 집단 간 비교가 아니므로 등분산 검정을 할 필요가 없다.
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 실습) 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게 변화

baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]
# 귀무가설 : 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게는 변화에 차이가 없다.
# 대립가설 : 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게는 변화에 차이가 있다.
print(np.mean(baseline))
print(np.mean(follow_up))

plt.bar(np.arange(2), [np.mean(baseline), np.mean(follow_up)])
# plt.show()

pair_sample = stats.ttest_rel(baseline, follow_up)
print('t-value : %.5f, p-value : %.5f'%pair_sample) # t-value : 3.66812, p-value : 0.00633
# 판정 : p-value : 0.00633 < 0.05 이므로 귀무가설을 기각한다. 즉, 복부 수술 전 9명의 몸무게와 복부 수술 후 몸무게는 변화에 차이가 있다.
