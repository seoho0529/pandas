# 웹 문서 읽기 : XML
import xml.etree.ElementTree as etree
"""
xmlf = open("../testdata/my.xml", mode="r", encoding='utf-8').read()
print(xmlf, type(xmlf))  # <class 'str'>
root = etree.fromstring(xmlf)
print(root, type(root))  # <class 'xml.etree.ElementTree.Element'> --> 파싱됨
print(root.tag, ' ', len(root))
print()
"""
xmlfile = etree.parse("../testdata/my.xml")
print(xmlfile, type(xmlfile))  # <class 'xml.etree.ElementTree.ElementTree'>
root = xmlfile.getroot()
print(root.tag) # items
print(root[0].tag)  # root의 0번쨰 태그인 item이 출력
print(root[0][0].tag) # name
print(root[0][1].tag) # tel
print(root[0][0].attrib) # {'id': 'ks1'}
print(root[0][0].attrib.keys())   # dict_keys(['id'])
print(root[0][0].attrib.values()) # dict_values(['ks1'])
print()
myname = root.find('item').find('name').text # root의 하위 요소 item 찾기.item의 하위요소 name찾기
print(myname)
print()
for child in root:
    # print(child.tag)
    for child2 in child:
        print(child2.tag, child2.attrib)
        
print()
children = root.findall('item')
print('children')
for it in children:
    re_id = it.find('name').get('id')
    re_name = it.find('name').text
    re_tel = it.find('tel').text
    print(re_id,re_name,re_tel)