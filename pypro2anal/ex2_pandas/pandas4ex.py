"""
import pandas as pd
#read
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv')
print(df)

# cut
bins = [1, 20, 35, 60, 150]
result_cut = pd.cut(df.Age, bins, labels=['소년','청년','장년','노년'])
print(result_cut)
print(df.groupby('Survived').count())
print(pd.value_counts(result_cut))

# pivot_table
print(df.pivot_table(values='Survived',index=['Sex'],columns=['Pclass']))

"""
import numpy as np
import pandas as pd
titanic = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv')
print(titanic['Age'])

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]


titanic.Age = pd.cut(titanic['Age'], bins=bins, labels=labels)
# group_survived = titanic.groupby(Age_cut)['Survived'].sum()
# print(group_survived)

print((titanic.pivot_table(values='Survived', index=['Sex'], columns=['Pclass'],  aggfunc=np.mean)*100).round(2))

df2 = titanic.pivot_table(values='Survived', index=['Sex', 'Age'], fill_value=0, columns=['Pclass'],  aggfunc=np.mean)*100
print(round(df2,2))