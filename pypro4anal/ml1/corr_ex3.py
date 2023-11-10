# 공공데이터 중 출입국 관광 서비스 자료를 사용 - 국내 유료 관광지에 방문한 외국인 관광객 관련 자료
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
import json

# 산점도 차트 작성, 상관계수 반환용 함수
def makeScatterGraph(tour_table, all_table, tourpoint):
    # 계산할 관광지명에 해당되는 자료를 뽑아 tour에 저장하고, 외국인관광객 자료와 병합
    tour = tour_table[tour_table['resNm'] == tourpoint]
    print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    print(merge_table)
    
    # 시각화
    fig = plt.figure()
    fig.suptitle(tourpoint + '상관관계분석')
    
    plt.subplot(1,3,1) # 1행 3열의 1열
    plt.xlabel('중국인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    print('r1 = ', r1)
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'], merge_table['ForNum'], s=5, c='red')
    
    plt.subplot(1,3,2) # 1행 3열의 2열
    plt.xlabel('일본인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    print('r2 = ', r2)
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'], merge_table['ForNum'], s=5, c='green')
    
    plt.subplot(1,3,3) # 1행 3열의 3열
    plt.xlabel('미국인 입국수')
    plt.ylabel('외국인 입장객수')
    lamb3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    print('r3 = ', r3)
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['usa'], merge_table['ForNum'], s=5, c='blue')
    
    plt.tight_layout()
    # plt.show() ##
    return [tourpoint, r1, r2, r3]  # tourpoint는 고궁명
    
    
def Chulbal():
    # json type의 관광지 정보 읽기
    fname = '../testdata/서울관광지입장정보.json'
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm','resNm','ForNum')) # 년월,관광지명,입장객수
    tour_table = tour_table.set_index('yyyymm') # 열에 있던 'yyyymm'이 인덱스로
    # print(tour_table)
    resNm = tour_table.resNm.unique() # 관광지명들을 중복된것 없이 보기
    # print('관광지명 : ', resNm) 
    print('관광지명 : ', resNm[:5])     # ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']
    
    # 중국인
    cdf = '../testdata/중국인방문객.json'
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())
    china_table = pd.DataFrame(jdata, columns=('yyyymm','visit_cnt'))
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    print(china_table[:3])
    
    # 일본인
    jdf = '../testdata/일본인방문객.json'
    jdata = json.loads(open(jdf, 'r', encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata, columns=('yyyymm','visit_cnt'))
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    print(japan_table[:3])
    
    # 미국인
    udf = '../testdata/미국인방문객.json'
    jdata = json.loads(open(udf, 'r', encoding='utf-8').read())
    usa_table = pd.DataFrame(jdata, columns=('yyyymm','visit_cnt'))
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    print(usa_table[:3])
    
    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    print(all_table.shape)
    
    r_list=[]  # r_list는 세 나라에 대한 상관계수를 가지고 있을 것.
    for tourPoint in resNm[:5]:
        r_list.append(makeScatterGraph(tour_table, all_table, tourPoint))
    
    r_df = pd.DataFrame(r_list, columns=('고궁명','중국','일본','미국'))
    r_df = r_df.set_index('고궁명') # 구조 재설정,,, 고궁명을 인덱스로
    print(r_df)
    r_df.plot(kind='bar', rot=50)
    #plt.show() ##
    
    
if __name__ == '__main__':
    Chulbal()
    
    

# 상관관계 문제)
# https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 
# 그리고 이들의 관계를 heatmap 그래프로 표현하시오. 
import seaborn as sns
df = pd.read_csv('../testdata/Advertising.csv')
print(df)
df1 = pd.DataFrame(df, columns=('tv','radio','newspaper'))
print(df1)
print(df1.corr())

sns.heatmap(df1.corr())
plt.show()
    
    
    
    