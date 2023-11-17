import pandas as pd
df = pd.read_csv('../testdata/winequality-red.csv')
print(df.head(3), df.shape)  # (1596, 12)

print(df.columns)

df_x = df.iloc[:-1]
print(df_x)

df_y = df['quality']
print(df_y)

from sklearn.preprocessing import LabelEncoder
df_y.loc[:,'quality'] = LabelEncoder().fit_transform(df_y['quality'])
