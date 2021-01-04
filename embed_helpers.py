# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:16:33 2020

@author: YF845WU
"""

from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import distance
# load the model
model = load_model('facenet_keras.h5')

#%%
embed_df = pd.read_pickle('./input/embeddings.pkl')
#%%
def faces_from_bytes(data_bytes) :
    inp = np.asarray(bytearray(data_bytes), dtype=np.uint8)

    image = cv2.imdecode(inp, cv2.IMREAD_COLOR)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )
    
    faces_in_image = []
    for (x, y, w, h) in faces:
        cropped_image = image[y:y + h, x:x + w]
        faces_in_image.append(cropped_image)
    return faces_in_image

def embed_from_array(pic_array):
    image = Image.fromarray(pic_array)
	
    image = image.convert('RGB')
    image = image.resize((160, 160))	
    pixels = np.asarray(image)
    
    face_pixels = pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std    
    samples = np.expand_dims(face_pixels, axis=0)    
    
    yhat = model.predict(samples)
    # get embedding
    embedding = yhat[0]
    
    return embedding  

def embedding_to_character(face_embedding) :
    dist_list = []
    for embedding in embed_df['embedding']:
        dst = distance.euclidean(embedding, face_embedding)
        dist_list.append(dst)
    embed_df['distance'] = dist_list
    
    closest_char = embed_df[embed_df.distance == embed_df.distance.min()].reset_index()['name'][0]

    return closest_char

#%%
def bytes_to_character(data_bytes) :
    faces = faces_from_bytes(data_bytes)
    print('Faces found: ' + str(len(faces)))
    face_results = []
    for face in faces :
        print('Analyzing Face...')
        face_embedding = embed_from_array(face)
        closest_char = embedding_to_character(face_embedding)
        print('Closest match: ' + closest_char)
        face_result_dict = {}
        face_result_dict['bytes'] = face
        face_result_dict['character'] = closest_char
        
        face_results.append(face_result_dict)
        print('\n')
    return face_results
        


