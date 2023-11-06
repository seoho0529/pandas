from django.shortcuts import render
from proapp.models import Jikwon, Buser
from django.db.models import F
import pandas as pd
import numpy as np
from django.utils.timezone import now
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rc('font',family='malgun gothic')
plt.rcParams['axes.unicode_minus']=False

# Create your views here.
def mainfunc(request):
    # 1) 사번, 직원명, 부서명, 직급, 연봉, 근무년수를 DataFrame에 기억 후 출력하시오. (join)
    year = Jikwon.objects.annotate(근무년수=now().year-F('jikwon_ibsail__year')).order_by('buser_num','jikwon_name')
    datas = year.values('jikwon_no', 'jikwon_name', 'buser_num__buser_name', 'jikwon_jik', 'jikwon_pay', '근무년수')
    df = pd.DataFrame(datas)
    df.columns=['사번','직원명','부서명','직급','연봉','근무년수']

    #  2) 부서명, 직급 자료를 이용하여  각각 연봉합, 연봉평균을 구하시오.
    b_paysum=df.pivot_table(index=['부서명'], values=['연봉'],aggfunc=sum)
    b_paymean=df.pivot_table(index=['부서명'], values=['연봉'],aggfunc=np.mean)
    j_paysum=df.pivot_table(index=['직급'], values=['연봉'],aggfunc=sum)
    j_paymean=df.pivot_table(index=['직급'], values=['연봉'],aggfunc=np.mean)
    
    # 3) 부서명별 연봉합, 평균을 이용하여 세로막대 그래프를 출력하시오.
    df2 = df['연봉'].groupby(df['부서명'])
    df2 = {'sum':df2.sum(), 'avg':df2.mean()}
    df3 = pd.DataFrame(df2)
    
    bu_result = df3.agg(['sum', 'mean'])
    bu_result.plot.bar()
    plt.title("부서별 급여 합/평균")
    fig = plt.gcf()
    fig.savefig('django_use1_ex/proapp/static/images/buserpay.png')
    
    # 4) 성별, 직급별 빈도표를 출력하시오.
    datas = year.values('jikwon_jik', 'jikwon_gen')
    df5 = pd.DataFrame(datas)
    df5.transpose()
    ctab = pd.crosstab(df5['jikwon_gen'],df5['jikwon_jik'], margins=True)
    

    return render(request, 'a.html', {'df':df.to_html(),'b_paysum':b_paysum.to_html(),
                  'b_paymean':b_paymean.to_html(),'j_paysum':j_paysum.to_html(),'j_paymean':j_paymean.to_html(),
                  'ctab':ctab.to_html()})




