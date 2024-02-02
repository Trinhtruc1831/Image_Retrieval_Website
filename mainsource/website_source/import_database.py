# import_database.py
import csv
from datetime import datetime
from search.models import Zoo # Replace 'myapp' with your actual app name
def import_database(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            Zoo.objects.create(
                image_name = row['image_name'],
                comment = row[' comment'],
                text_embeddings = row['text_embeddings'],
                img_embeddings = row['img_embeddings'],
                cos_sim =row['cos_sim'],
            )
if __name__ == '__main__':
    csv_file_path = '../resource/animal_text_img_dataset_extracted.csv' # Replace with your actual file path
    import_database(csv_file_path)


  