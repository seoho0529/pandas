# Selenium 툴을 이용해 브라우저를 통제. 웹 크롤링 가능
from selenium import webdriver
import time

"""
from selenium import webdriver
browser = webdriver.Chrome()    #<== 브라우저 열기
browser.implicitly_wait(5)                 #<== 선택적인 명령으로 지정한 시간(초)동안 기다린다.
browser.get('https://daum.net')         # <== 원하는 url을 적어 줌. 해당 사이트가 열린다.
browser.quit()                               # <== 모든 작업을 끝내고 브라우저를 닫음
"""

"""
from selenium import webdriver
browser = webdriver.Chrome()  #Optional argument, if not specified will search path.
browser.get('http://www.google.com/xhtml');
search_box = browser.find_element("name", "q")
search_box.send_keys('파이썬')
search_box.submit()
time.sleep(5)          # Let the user actually see something!
browser.quit()
"""

try:
    url = "https://www.daum.net"
    browser = webdriver.Chrome()
    browser.implicitly_wait(3)
    browser.get(url);
    browser.save_screenshot("daum_img.png")
    browser.quit()
    print('성공')
    
except Exception:
    print('에러')