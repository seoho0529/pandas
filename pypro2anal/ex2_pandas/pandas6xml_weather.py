# https://www.kma.go.kr/XML/weather/sfc_web_map.xml
# 기상청 제공 국내 주요 지역 날씨정보 XML 문서 읽기

import urllib.request
import xml.etree.ElementTree as etree

# 문서를 읽어 파일로 저장 후 xml 문서 처리
try:
    webdata = urllib.request.urlopen('https://www.kma.go.kr/XML/weather/sfc_web_map.xml')
    print(webdata) # HTTPResponse object
    webxml = webdata.read()
    webxml = webxml.decode('utf-8')
    #print(webdata)
    webdata.close()
    with open('pandas6.xml', mode='w', encoding='utf-8') as obj:
        obj.write(webxml)
    print("성공이야~!")
except Exception as e:
    print('err : ',e)

xmlfile = etree.parse("pandas6.xml")
print(xmlfile)     # ElementTree object
root = xmlfile.getroot()
print(root.tag)    # {current}current
print(root[0].tag) # {current}weather

children = root.findall('{current}weather')
print(children) # Element

for child in children:
    y = child.get('year')   # 속성 값 읽기
    m = child.get('month')
    d = child.get('day')
    h = child.get('hour')
print(y+'년' + m + "월" + d + "일" + h + "시 현재 날씨 정보")
    
datas = []
for child in root:
    # print(child.tag)   # {current}weather
    for ch in child:
        # print(ch.tag)    # {current}local
        localName = ch.text  # 지역명 localName (ex.속초.동두천 ...)
        re_ta = ch.get("ta") # 속성 값 얻기 (19.2 20.7 ,,,)
        re_desc = ch.get('desc')
        print(localName, re_ta, re_desc) # 동두천 18.8 맑음 ...
        datas += [[localName, re_ta, re_desc]]

# print(datas)  # [['속초', '20.7', '맑음'], ['북춘천', '19.8', '맑음'], ['철원', '18.7', '구름조금'],
import pandas as pd
import numpy as np

df = pd.DataFrame(datas, columns=['지역','온도','상태'])
print(df.head(3))
print(df.tail(3))

imsi = np.array(df.온도)
imsi = list(map(float, imsi))
print(imsi) # ['20.7' '19.8' '18.7'
print('평균온도 : ', round(np.mean(imsi),2))