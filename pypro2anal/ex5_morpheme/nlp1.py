# 한글 형태소 분석
# 형태소 : 
# 한글 형태소 분석 도구로 KoNLpy를 사용
from konlpy.tag import Kkma, Okt, Komoran

kkma = Kkma()
print(kkma.sentences('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 문장 나누기
print(kkma.pos('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # pos : 품사태깅
print(kkma.nouns('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 명사만
print(kkma.morphs('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.'))

print()
okt = Okt()
print(okt.phrases('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 어절 추출
print(okt.pos('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # pos : 품사태깅
print(okt.pos('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.', stem=True)) # pos : 품사태깅,, stem=True이면 어근으로 출력
print(okt.nouns('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 명사만
print(okt.morphs('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.'))

print()
ko = Komoran()
print(ko.pos('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # pos : 품사태깅
print(ko.nouns('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 명사만
print(okt.morphs('시속 200㎞에 달하는 강풍을 동반한 태풍 키아란이다. 서유럽을 강타하며 최소 7명이 사망했다.')) # 텍스트에서 형태소를 반환한다
