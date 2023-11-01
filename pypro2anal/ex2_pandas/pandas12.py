# 기상청 제공 중기 예보 웹 문서 읽기
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url="https://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
data = urllib.request.urlopen(url).read()
# print(data)

soup = BeautifulSoup(data, 'lxml')
title = soup.find('title').string
print(title)
# wf = soup.find('wf')
# print(wf.text)

city = soup.find_all('city')
# print(city)
cityDatas = []
for c in city:
    cityDatas.append(c.string)

df = pd.DataFrame()
df['city'] = cityDatas  # df에 city열을 만들고 값 넣어줌
# print(df)

# next sibling은 +로 기입, previous sibling은 -로 기입
tempMins = soup.select('location > province + city + data > tmn')  # 대부분 태그들은 소문자가 기본
tempMaxs = soup.select('location > province + city + data > tmx')
# print(tempMins)
tempDatas1 = []
for t in tempMins:
    tempDatas1.append(t.string)

tempDatas2 = []
for t in tempMaxs:
    tempDatas2.append(t.string)

df['temp_m'] = tempDatas1
df['temp_x'] = tempDatas2
df.columns = ['지역','최저기온','최고기온']
print(df)