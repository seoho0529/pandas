# pandas 문제 4)
import pandas as pd

df = pd.read_csv('../testdata/human.csv')
print(df.head(3))
print()

print('strip() 함수를 사용하여 공백 제거')
df.columns = df.columns.str.strip()
print(df.head(3))
print()

print('Group인 NA인 행 삭제')
df = df[df['Group'].str.strip() != 'NA']
print(df.head(5))
print()


print('Career, Score 칼럼을 추출하여 DataFrame 을 작성')
df = df[['Career', 'Score']]
print(df)
print()

print('Career, Score 칼럼의 평균 계산')
avg_career = df['Career'].mean()
avg_score = df['Score'].mean()
print()

# 결과 출력
print(df)
print(f'평균 Career: {avg_career}')
print(f'평균 Score: {avg_score}')


# tips.csv 파일 읽기
tips = pd.read_csv('../testdata/tips.csv')
print(tips.head(3))

# 파일 정보 확인
print("파일 정보 확인:")
print(tips.info())
print()

# 앞에서 3개의 행 출력
print("\n앞에서 3개의 행 출력:")
print(tips.head(3))
print()

# 요약 통계량 보기
print("\n요약 통계량 보기:")
print(tips.describe())
print()

# 흡연자, 비흡연자 수 계산
smoke_counts = tips['smoker'].value_counts()
print("\n흡연자, 비흡연자 수:")
print(smoke_counts)
print()


# 요일 칼럼의 유일한 값 출력
uni_days = tips['day'].unique()  # pd.unique(tips['day'])
print("\n요일 칼럼의 유일한 값:")
print(uni_days)
