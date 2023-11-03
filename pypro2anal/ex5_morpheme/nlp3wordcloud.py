# 웹 문서에서 검색 자료를 읽어 형태소 분석 후 명사만 추출한 다음 워드클라우드 차트 그리기

# pip install pygame, pip install simplejson, pip install pytagcloud
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

# keyword = input('검색어 : ')
keyword='단풍'
print(quote(keyword)) # 인코딩

# 신문사 검색 기능을 이용
targetUrl = "https://www.donga.com/news/search?query=" + quote(keyword)
print(targetUrl)

source = urllib.request.urlopen(targetUrl)

soup = BeautifulSoup(source, 'lxml')
print(soup)

#content > div.sch_cont > div:nth-child(7) > div > div:nth-child(2) > div.rightList > span.txt > a
msg=""
for title in soup.select('div.rightList > span.txt'):
    title_link = title.find('a')
    # print(title_link)
    articleUrl = title_link['href']  # title_link의 href 속성
    # print(articleUrl)
    try:
        sourceArticle = urllib.request.urlopen(articleUrl)  # 실제 기사 내용 페이지로 접근
        soup = BeautifulSoup(sourceArticle, 'lxml', from_encoding='utf-8')
        contents = soup.select('div.article_txt')
        for imsi in contents:
            item = str(imsi.find_all(string=True))
            #print(item)
            msg = msg + item
    except Exception as e:
        pass  # 내용이 없는 것을 지나치기 위해 pass

print(msg)

from konlpy.tag import Okt
from collections import Counter

okt = Okt()
nouns = okt.nouns(msg)
print(nouns)

result = [] # 두 글자 이상만 저장할 result
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)
        
print(result)

count = Counter(result)
print(count)
tag = count.most_common(60)  # 상위 60개만 참여
print(tag) # 출력형태 : [('한국', 24), ('코스', 16), ...  | 이런 형태를 pytagcloud가 선호함

import pytagcloud
taglist = pytagcloud.make_tags(tag, maxsize=100)
print(taglist)  # [{'color': (80, 190, 157), 'size': 115, 'tag': '한국'},

pytagcloud.create_tag_image(taglist, output='word.png', size=(1000,600), background=(0,0,0),
                            fontname="Korean", rectangular=True)

# 저장된 이미지 읽기
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('word.png')
plt.imshow(img)
plt.show()

import webbrowser
webbrowser.open('word.png')



    