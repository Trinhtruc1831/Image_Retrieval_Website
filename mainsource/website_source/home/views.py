from django.http import HttpResponse
from django.template import loader

def index(request):
    template = loader.get_template('homepage.html')
   
    context = {
        'rs': "",
    }
    
    return HttpResponse(template.render(context, request))
