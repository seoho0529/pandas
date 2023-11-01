# JSON 문서 읽기 : 서울시 제공 도서관 정보 5개 읽기
import json
import urllib.request as req

url="https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.json"
plainText = req.urlopen(url).read()  # 안되면 .decode()까지 해줘도 됨

jsonData = json.loads(plainText)  # json decoding   |   str -> dict 로 바꿈
print(jsonData)
print(type(jsonData))  # <class 'dict'>

libData = jsonData.get('SeoulLibraryTime').get('row')  # 도서관 이름만 가져오기
print(libData)
name = libData[0].get('LBRRY_NAME')
print(name)  # 한개 출력이 잘 되니 for문으로,,

print('\n도서관\t전화\t주소\n')
datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    print(name + "\t" + tel + "\t" + addr)
    imsi = [name, tel, addr]
    datas.append(imsi)

import pandas as pd
df = pd.DataFrame(datas, columns=['이름', '전화', '주소'])
print(df)