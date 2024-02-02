from django.db import models
import django

# django.setup()

class Zoo(models.Model):
  image_name = models.CharField(max_length=20)
  comment = models.CharField(max_length=255)
  text_embeddings = models.TextField()
  img_embeddings = models.TextField()
  cos_sim = models.FloatField()
