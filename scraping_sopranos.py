# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:36:50 2020

@author: YF845WU
"""

from google_images_download import google_images_download   #importing the library
import os
response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":'tony soprano',"limit":3,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images

#%%
names_main = ['tony soprano','christopher moltisanti','paulie gualtieri','silvio dante',
         'carmela soprano','adriana la cerva','jennifer melfi']
#%%
names_extended = ['tony soprano','christopher moltisanti','paulie gualtieri','silvio dante',
         'carmela soprano','adriana la cerva','jennifer melfi','johnny sack','livia soprano',
         'junior soprano','meadow soprano','aj soprano','artie bucco','charmaine bucco',
         'phil leotardo','janice soprano','bobby bacala','tony blundetto','pussy sopranos',
         'furio sopranos','ralph cifaretto','vito spatafore','richie aprile','rosalie aprile']

#%%
def get_pictures(name, limit = 10):
    response = google_images_download.googleimagesdownload()
    
    arguments = {"keywords":name,"limit":limit,"print_urls":True}
    paths = response.download(arguments)
    
    pic_list = []
    char = name
    for path in paths[0][char]:
        char_dict = {}
        char_dict['name'] = char
        char_dict['path'] = path
        pic_list.append(char_dict)
    return pic_list
#%%
test_pic_list = get_pictures('griffin smith')
#%%
for entry in test_pic_list :
    print(entry)
    print('\n')
#%%
master_pic_list = []
for name in names_extended :
    pic_list = get_pictures(name, 7)
    for entry in pic_list :
        master_pic_list.append(entry)

    
    
    
#%%
os.mkdir('./faces/test')

#%%
import cv2
import sys
import numpy as np
#%%
name = 'adriana la cerva'
folder_path = './downloads/' + name + '/'
file_name = '1.340.jpg'
file_path = folder_path + file_name
extension = file_name[-4:]
#%%
image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#%%
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
)    
#%%
os.mkdir('./faces/' + name)
#%%
new_file_path = './faces/' + name + '/' + file_name + '_face'
#%%
i = 1
for (x, y, w, h) in faces:
    cropped_image = image[y:y + h, x:x + w]
    cv2.imwrite(new_file_path + str(i) + extension, cropped_image)
    
#%%
def get_faces(name, file_name) :
    face_list = []
    folder_path = './downloads/' + name + '/'
    file_path = folder_path + file_name
    extension = file_name[-4:]
    
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
            )
    
    new_file_path = './faces/' + name + '/' + file_name + '_face'
    i = 1
    for (x, y, w, h) in faces:
        cropped_image = image[y:y + h, x:x + w]
        new_image_path = new_file_path + str(i) + extension
        cv2.imwrite(new_image_path, cropped_image)
        face_dict = {}
        face_dict['name'] = name
        face_dict['path'] = new_image_path
        face_list.append(face_dict)
        
    return face_list
#%%
test_face = get_faces('adriana la cerva', '3.895000dd541966946d1fab87cc81c39b.png')

#%%
list_files = os.listdir('./downloads/' + name)
master_face_list = []
for name in names_extended :
    print(name)
    print('\n')
    list_files = os.listdir('./downloads/' + name)
    try:
        os.mkdir('./faces/' + name)
        print('directory made')
    except :
        print('directory already made')
    for file in list_files :
        print(file)
        face_list = get_faces(name, file)
        for face_dict in face_list:
            master_face_list.append(face_dict)
            
#%%
from keras.models import load_model
from PIL import Image
from numpy import asarray
import numpy as np
# load the model
model = load_model('facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)
#%%
name = 'artie bucco'
folder_path = './input/faces/' + name + '/'
file_name = '5.c77b73420a3274d07befc3e729ef50b3.jpg_face1.jpg'
file_path = folder_path + file_name

image = Image.open(file_path)
	
image = image.convert('RGB')
image = image.resize((160, 160))	
pixels = asarray(image)
#%%
face_pixels = pixels.astype('float32')
# standardize pixel values across channels (global)
mean, std = face_pixels.mean(), face_pixels.std()
face_pixels = (face_pixels - mean) / std    
samples = np.expand_dims(face_pixels, axis=0)    
#%%
yhat = model.predict(samples)
# get embedding
embedding_artie1 = yhat[0]    
    
#%%
from scipy.spatial import distance
#%%
dst = distance.euclidean(embedding_adriana2, embedding_artie1)
print(dst)

#%%
def embed_from_path(file_path):
    image = Image.open(file_path)
	
    image = image.convert('RGB')
    image = image.resize((160, 160))	
    pixels = asarray(image)
    
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
import os
master_embed_list = []
for name in names_extended :
    print(name)
    print('\n')
    list_files = os.listdir('./input/faces/' + name)
    for file in list_files :
        print(file)
        folder_path = './input/faces/' + name + '/'
        file_path = folder_path + file
        embedding = embed_from_path(file_path)
        embed_dict = {}
        embed_dict['name'] = name
        embed_dict['file_path'] = file_path
        embed_dict['embedding'] = embedding
        master_embed_list.append(embed_dict)
        print('\n')

#%%
import pandas as pd
embed_df = pd.DataFrame(master_embed_list)
embed_df.to_pickle('./input/embeddings.pkl')
#%%
embed_to_compare = embed_df['embedding'][6]  
#%%
dist_list = []
for embedding in embed_df['embedding']:
    dst = distance.euclidean(embedding, embed_to_compare)
    dist_list.append(dst)
embed_df['distance'] = dist_list
    
#%%

    
    
    
    
    
    
    
    
    
    
    
    
    