# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:35:30 2020

@author: YF845WU
"""

import embed_helpers as eh
import cv2
import numpy as np
from PIL import Image

#%%
test_embed = eh.embed_from_path('./input/test/')


#%%
file_1 = './input/test/220px-Edie_Falco_2010.jpg'
with open(file_1, "rb") as f:
        data_bytes_1 = f.read()
#%%
inp = np.asarray(bytearray(data_bytes_1), dtype=np.uint8)

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
#%%
test_faces = faces_from_bytes(data_bytes_1)
pil_img = Image.fromarray(faces_in_image[0])
#%%
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

#%%  
test_embed = embed_from_array(faces_in_image[0])

#%%
dist_list = []
for embedding in embed_df['embedding']:
    dst = distance.euclidean(embedding, test_embed)
    dist_list.append(dst)
embed_df['distance'] = dist_list

#%%
closest_char = embed_df[embed_df.distance == embed_df.distance.min()].reset_index()['name'][0]

#%%
def embedding_to_character(face_embedding) :
    dist_list = []
    for embedding in embed_df['embedding']:
        dst = distance.euclidean(embedding, face_embedding)
        dist_list.append(dst)
    embed_df['distance'] = dist_list
    
    closest_char = embed_df[embed_df.distance == embed_df.distance.min()].reset_index()['name'][0]

    return closest_char

#%%
test_e2c = embedding_to_character(test_embed)
#%%
import embed_helpers as eh
import base64
import cv2
#%%
file_1 = './input/test/MV5BMTkwMDY4NDMzN15BMl5BanBnXkFtZTcwNDkzNDE5NQ@@._V1_.jpg'
with open(file_1, "rb") as f:
        data_bytes_1 = f.read()
#%%
test_e2e = eh.bytes_to_character(data_bytes_1)
        
result_bytes = cv2.imencode('.jpg', test_e2e[0]['bytes'])[1].tobytes()
image = base64.b64encode(result_bytes).decode("utf-8")        
        
        
        
        
        
        
        
        
        
        