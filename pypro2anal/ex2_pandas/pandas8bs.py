# BeautifulSoup의 fine, select 연습
from bs4 import BeautifulSoup
htmlData = """
<html>
<body>
<h1>제목 태그</h1>
<p>문단 1</p>
<p>문단 2</p>
</body>
</html>
"""
print(type(htmlData))   # <class 'str'>
soup = BeautifulSoup(htmlData, 'html.parser') # BeautifulSoup 객체 생성 - BeautifulSoup 모듈 지원 명령 사용 가능
print(type(soup))       # <class 'bs4.BeautifulSoup'>
h1 = soup.html.body.h1
print('h1 : ', h1.text, ' ', h1.strig)      # h1 :  제목 태그   None
p1 = soup.html.body.p
print('p1 :', p1.text)  # p : 문단 1

p2 = p1.next_sibling.next_sibling
print('p2 :', p2.text)

print('\n-------- find함수 : 반환값이 1개 사용 --------')
htmlData2 = """
<html>
<body>
<h1 id="title">제목 태그</h1>
<p>문단 1</p>
<p id="my" class="our">문단 2</p>
</body>
</html>
"""

soup2 = BeautifulSoup(htmlData2, 'html.parser')
print(soup2.p, ' ', soup2.p.string)
print(soup2.find('p').string)
print(soup2.find('p', id='my').string)
print(soup2.find(id='title').string)
print(soup2.find(id='my').string)
print(soup2.find(class_='our').string)
print(soup2.find(attrs={'class':'our'}).string)
print(soup2.find(attrs={'id':'my'}).string)


print('\n-------- find_all(), findAll() 함수 : 반환값이 여러개 사용 --------')
htmlData3 = """
<html>
<body>
<h1 id="title">제목 태그</h1>
<p>문단 1</p>
<p id="my" class="our">문단 2</p>
<div>
    <a href="https:www.naver.com">naver</a><br/>
    v<a href="https:www.daum.net">daum</a>
</div>
</body>
</html>
"""
soup3 = BeautifulSoup(htmlData3, 'lxml')
print(soup3.find_all("a"))
print(soup3.find_all("a","p"))
print(soup3.findAll(["a"]))
print()
links = soup3.findAll(["a"])
print(links)
for i in links:
    href = i.attrs['href']
    text = i.string
    print(href, ' ', text)

print('정규 표현식 사용')
import re
links2 = soup3.find_all(href=re.compile(r'^https'))
for j in links2:
    print(j.attrs['href'])
    
""" 
print('벅스 뮤직 사이트에서 곡 제목 읽기')
from urllib.request import urlopen
url = urlopen("https://music.bugs.co.kr/chart")
soup = BeautifulSoup(url.read(), 'html.parser')
# print(soup)
musics = soup.find_all('td', class_='check')
# print(musics)
for i, music in enumerate(musics):
    print("{}위 : {}".format(i+1, music.input['title']))
"""

# 자손인지 직계인지, 클래스인지 id인지 
print('\n-------- select_one(find), select(findall)() 함수 : css의 셀렉터를 사용 --------')
htmlData4 = """
<html>
<body>
<div id="hello">
    <a href="https:www.naver.com">naver</a><br/>
    <span>
        <a href="https:www.daum.net">daum</a>
    </span>
    <b>
        <ul class="world">
            <li>안녕</li>
            <li>반가워</li>
        </ul>
    </b>
</div>
<div id="hi" class="good">두번쨰 디브 태그</div>
</body>
</html>
"""
soup4 = BeautifulSoup(htmlData4, 'lxml')
kbs = soup4.select_one("div#hello > a")  # 단수 선택
print('kbs : ', kbs, ' ', kbs.string)
kbs2 = soup4.select_one("div.good")
print('kbs2 : ', kbs2, ' ', kbs2.string)
print()
mbc = soup4.select("div#hello ul.world > li") # 복수 선택  div태그 중에서 id가 hello인,, 그게 ul를 포함하고 있음 -- 자손을 뜻함(공백유무),, hello>ul.world는 직계를 뜻함
print('mbc : ', mbc)
for a in mbc:
    print(a.string, ' ')

print()
msg = list()
for a in mbc:
    msg.append(a.string)
    
import pandas as pd
df = pd.DataFrame(msg, columns=['자료'])
print(df)
print(df.to_json())