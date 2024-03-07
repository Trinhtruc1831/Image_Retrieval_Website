# import os
# import torch
# import skimage
# import requests
import numpy as np
import pandas as pd

# from io import BytesIO
# import IPython.display
# import matplotlib.pyplot as plt
# from datasets import load_dataset
# from collections import OrderedDict
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import haversine_distances

from .models import Zoo

def get_single_text_embedding(text):
  device =  "cpu"

  model_ID = "openai/clip-vit-base-patch32"
   # Save the model to device
  model = CLIPModel.from_pretrained(model_ID).to(device)

  # Get the tokenizer
  tokenizer = CLIPTokenizer.from_pretrained(model_ID)

  inputs = tokenizer(text, return_tensors = "pt").to(device)
  text_embeddings = model.get_text_features(**inputs)

  # convert the embeddings to numpy array
  embedding_as_np = text_embeddings.cpu().detach().numpy()

  return embedding_as_np

def get_single_image_embedding(my_image):
  
  device =  "cpu"

  model_ID = "openai/clip-vit-base-patch32"
   # Save the model to device
  model = CLIPModel.from_pretrained(model_ID).to(device)

  # Get the processor
  processor = CLIPProcessor.from_pretrained(model_ID)

  image = processor(
      text = None,
      images = my_image,
      return_tensors="pt"
  )["pixel_values"].to(device)

  embedding = model.get_image_features(image)

  # convert the embeddings to numpy array
  embedding_as_np = embedding.cpu().detach().numpy()

  return embedding_as_np

def parse_embedding_array(str_embedding):
    str_embedding  = str_embedding.replace('\n', '')
    str_embedding = str_embedding.replace('[[','')
    str_embedding = str_embedding.replace(']]','')
    array = np.array(str_embedding.split(" "))
    array = np.delete(array, np.where(array == "")).reshape(1,512)
    return array
    

def get_top_N_images(query, top_K=100, search_criterion="text"):

    data = pd.DataFrame(
        list(
            Zoo.objects.all().values()
        )
    )

    # """
    # Retrieve top_K (5 is default value) articles similar to the query
    # """
    # Text to image Search
    if(search_criterion.lower() == "text"):
      query_vect = get_single_text_embedding(query)

    # # Image to image Search
    if(search_criterion.lower() == "img"):
      query_vect = get_single_image_embedding(query)

    # Relevant columns
    revevant_cols = ["image_name","comment", "cos_sim"]

    type_of_similarity = "dot_product" #"dot_product" "euclidean_distances" "cosine_similarity"

    if type_of_similarity == "cosine_similarity":
      print("🌽")
      data["similarity"] = data["img_embeddings"].apply(lambda x: cosine_similarity(query_vect, parse_embedding_array(x)))
      data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

      most_similar_articles = data.sort_values(by='similarity', ascending=False).drop_duplicates(subset='image_name')[1:top_K+1]
    
    if type_of_similarity == "euclidean_distances":
      print("🍒")
      data["similarity"] = data["img_embeddings"].apply(lambda x: euclidean_distances(query_vect, parse_embedding_array(x)))
      data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

      most_similar_articles = data.sort_values(by='similarity', ascending=True).drop_duplicates(subset='image_name')[1:top_K+1]
    
    if type_of_similarity == "dot_product":
      data["similarity"] = data["img_embeddings"].apply(lambda x: np.dot(np.array(query_vect.reshape(512),dtype=float), np.array(parse_embedding_array(x).reshape(512),dtype=float)))
      most_similar_articles = data.sort_values(by='similarity', ascending=False).drop_duplicates(subset='image_name')[1:top_K+1]

    print(data.sort_values(by='similarity', ascending=False)["similarity"].values)
    return most_similar_articles[revevant_cols].reset_index()