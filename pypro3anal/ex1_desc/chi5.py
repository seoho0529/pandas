# 이원카이제곱 검정
import pandas as pd
import scipy.stats as stats

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/home_task.txt", \
                   sep='\t', index_col=0)
print(data.head(3))

# 귀무가설 : 집안일의 종류와 일하는 사람은 관계가 없다.(독립적이다)
# 대립가설 : 집안일의 종류와 일하는 사람은 관계가 있다.(독립적이지 않다)

chi2, pvalue, _, _ = stats.chi2_contingency(data)
print('chi2:{}, pvalue:{}'.format(chi2, pvalue))
# chi2:1364.5404438935336, pvalue:1.8759478966116962e-273
# pvalue가 0.05보다 작으므로 귀무가설 기각, 대립가설을 채택한다. 즉 집안일의 종류와 일하는 사람은 관계가 있을 것이다.