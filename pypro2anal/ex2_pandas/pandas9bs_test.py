
# 웹사이트 스크래핑 연습
import urllib.request as req
from bs4 import BeautifulSoup
import urllib
 
print('연습1 : 위키백과에서 검색된 자료 읽기-----------')
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)
#print(wiki.read())
soup = BeautifulSoup(wiki, 'html.parser')  # 'html.parser' or 'lxml' 사용을 추천
print(soup.select("#mw-content-text > div.mw-parser-output > p"))
print()
result = soup.select("div.mw-parser-output > p > b")  # <b>태그 걸린거 잡기
print(result)

for s in result:
    print(s.string)
    
    
print('\n연습2 : 다음에서 뉴스 자료 읽기 -----------')
url = "https://news.daum.net/society#1"
daum = req.urlopen(url)
soup = BeautifulSoup(daum, 'lxml')
data = soup.select_one("body > div.direct-link > a")  # body의 자식으로 div가 있고 그 밑의 a
print(data)  # <a href="#mainContent">본문 바로가기</a>

datas = soup.select("div.direct-link > a")  # body의 자식으로 div가 있고 클래스명이 direct-link이며 그 밑의 a
print(datas) # [<a href="#mainContent">본문 바로가기</a>, <a href="#gnbContent">메뉴 바로가기</a>]

for i in datas:
    href=i.attrs['href']
    text = i.string
    print('href:%s, text:%s'%(href, text))

print()
datas = soup.findAll('a')
# print(datas)
for i in datas:
    href=i.attrs['href']
    text = i.string
    print('href:%s, text:%s'%(href, text))  # a태그 모두 출력
    
print('\n연습2 : 네이버에서 시장지표 자료 중 미국USD자료 읽기 (일정 시간마다 주기적으로 읽어 파일로 저장) -----------')
import datetime
import time

def workingFunc():
    url = "https://finance.naver.com/marketindex/"
    data = req.urlopen(url)
    soup = BeautifulSoup(data, 'lxml')
    price = soup.select_one("div.head_info > span.value").string
    # print('미국 USD : ', price)
    t = datetime.datetime.now()
    fname = "./usd/" + t.strftime('%Y-%m-%d-%H-%M-%S') + '.txt'
    
    with open(fname, mode='w') as f:
        f.write(price)
"""
while True:  # 몇일동안 돌릴지 날짜를 지정해줄 수도 있음
    workingFunc()
    time.sleep(3)  # 계속 읽으면 안되니 시간 정해주기
""" 

print('교촌치킨 메뉴와 가격 읽기 ----------')
url = "https://www.kyochon.com/menu/chicken.asp"
ck = req.urlopen(url)
soup = BeautifulSoup(ck, 'html.parser')
data1 = soup.select('dl.txt > dt')
data2 = soup.select('p.money > strong')
#print(data1)
#print(data2)
"""
for i in data1:
    text = i.string
    print('name : %s'%(text))

for i in data2:
    text = i.string
    print('price : %s'%(text))
"""
for i1, i2 in zip(data1, data2):
    text1 = i1.string
    text2 = i2.string
    #print('name : %s' % text1)
    #print('price : %s' % text2)
    print('%s : %s'%(text1, text2))
    

name = soup.select("div.tabConts > ul.menuProduct > li")    
"""
max_price = None
min_price = None
for item in name:
    mname = item.select_one("dl > dt").text
    price = int(item.select_one("p.money > strong").text.replace(",", ""))
    # 총 가격 계산
    if max_price is None or price > max_price:
        max_price = price
        max_menu = mname
print('제일 비싼 메뉴는',max_menu,':',max_price)

for item in name:
    mname = item.select_one("dl > dt").text
    price = int(item.select_one("p.money > strong").text.replace(",", ""))
    # 총 가격 계산
    if min_price is None or price < min_price:
        min_price = price
        min_menu = mname
print('제일 싼 메뉴는',min_menu,':',min_price)
"""
max_price = None
min_price = None


for item in name:
    mname = item.select_one("dl > dt").text
    price = int(item.select_one("p.money > strong").text.replace(",", ""))
    
    if max_price is None or price > max_price:
        max_price = price
        max_menu = mname

    if min_price is None or price < min_price:
        min_price = price
        min_menu = mname


print('제일 비싼 메뉴는', max_menu, ':', max_price,'원')
print('제일 싼 메뉴는', min_menu, ':', min_price,'원')
