# local DB (sqlite)와 연동 후 DataFrame에 저장
import sqlite3

sql = "create table if not exists mytab(product varchar(10), maker varchar(10), weight real, price integer)"
conn = sqlite3.connect('testdb')  # file로 저장.영구적
conn = sqlite3.connect(':memory:')  # 실행 시 램에 저장.휘발성(반영구적)
conn.execute(sql)
conn.commit()

data1 = ('마우스','삼성',12.5,7000)
data2 = ('키보드','엘지',82.5,17000)
stmt = 'insert into mytab values(?,?,?,?)'
conn.execute(stmt, data1)
conn.execute(stmt, data2)  # execute는 한개씩 수행, executemany는 여러개 수행

datas=[('신상1','삼성',1.5,1000),('신상2','삼성',1.5,2000),('신상3','삼성',1.5,3000)]
conn.executemany(stmt, datas)
conn.commit()

cursor = conn.execute("select * from mytab")
rows= cursor.fetchall()
print(rows[0])
print(rows[1])
for a in rows:
    print(a)
    
import pandas as pd
df1 = pd.DataFrame(rows, columns=['품명','제조사','무게','가격'])  # rows를 DataFrame타입으로 바꿔주고 칼럼명 설정
print(df1)
print('---'*10)
# 위보다 간편하게 sql문 바로 DataFrame 형식으로 바꾸기
df2 = pd.read_sql("select * from mytab", conn)
print(df2)
counts = pd.read_sql("select count(*) from mytab", conn)
print('레코드 건수 : ', counts)

# DataFrame의 자료를 table에 저장(insert)
mydata = {
    'product':['연필','볼펜','지우개'],
    'maker':['모나미','모나미','ain'],
    'weight':['2.2','3.4','5.0'],
    'price':['100','1200','600'],
}
frame = pd.DataFrame(mydata)
# print(frame)
frame.to_sql("mytab",conn, if_exists="append", index=False)
print('저장성공')
df3 = pd.read_sql("select * from mytab", conn)
print(df3)


# 참조 : DataFrame의 자료를 원격 table 저장하려면 pip install pymysql, pip install sqlalchemy
