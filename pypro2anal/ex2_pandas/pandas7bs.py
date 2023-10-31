# BeautifulSoup : HTML 및 XML 파일에서 데이터를 가져오는 Python 라이브러리
# 문서 내의 element(tag, 요소) 및 attribute(속성) 찾기 함수 : find(), find_all(), select_one(), select()
import requests  
from bs4 import BeautifulSoup

def go():
    base_url = "http://www.naver.com:80/index.html"
    #storing all the information including headers in the variable source code
    source_code = requests.get(base_url)
    print(source_code) # <Response [200]>

    #sort source code and store only the plaintext
    plain_text = source_code.text
    print(plain_text, type(plain_text)) # <class 'str'>
    
    # #converting plain_text to Beautiful Soup object so the library can sort thru it
    convert_data = BeautifulSoup(plain_text, 'lxml') # 'lxml'은 파서를 의미
    print(type(convert_data)) # <class 'bs4.BeautifulSoup'>
    
    for link in convert_data.findAll('a'):
        href = base_url + link.get('href')  #Building a clickable url
        print(href)                          #displaying href
go()