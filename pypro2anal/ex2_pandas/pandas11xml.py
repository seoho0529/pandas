# XML 문서 읽기 : 서울시 제공 도서관 정조 5개 읽기
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://raw.githubusercontent.com/pykwon/python/master/seoullibtime5.xml"
plainText = req.urlopen(url).read().decode()  # try~except 사용을 권장
print(plainText)

xmlobj = BeautifulSoup(plainText, 'lxml')
libData = xmlobj.select('row')
# print(libData)  # lbrry_name  adres

for d in libData:
    name = d.find('lbrry_name').text
    addr = d.find('adres').text
    print('도서관명 : ',name)
    print('주소 : ',addr + "\n")