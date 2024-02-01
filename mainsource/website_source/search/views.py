from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import zoo_data_embedding

def index(request):
    # csv_file = request.FILES['csv_file'].read().decode('utf-8').splitlines()
    # csv_reader = csv.DictReader(csv_file)

    # for row in csv_reader:
    #     Book.objects.create(
    #         title=row['title'],
    #         author=row['author'],
    #         publication_year=row['publication_year'],
    #         isbn=row['isbn']
    #     )


    template = loader.get_template('searchview.html')
    return HttpResponse(template.render())