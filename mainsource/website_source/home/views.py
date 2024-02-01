from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("HOMEPAGE: This is the project của Mực Nhinh, Phuyên Ương, Thúc Tranh!")