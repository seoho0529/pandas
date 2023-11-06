from django.shortcuts import render

# Create your views here.
def surveyMain(request):
    return render(request, 'main.html')

def surveyView(request):
    return render(request, 'survey.html')

def surveyProcess(request):
    pass

def surveyAnalysis(request):
    pass