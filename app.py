# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:42:23 2020

@author: YF845WU
"""

from flask import Flask, request, jsonify, abort, render_template
import embed_helpers as eh
import base64
import cv2

app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def access_file():
   if request.method == 'POST':
      f = request.files['file']
      data = f.read()
      
      results = eh.bytes_to_character(data)
      if len(results) > 0 :
          
          character = results[0]['character']
          image_bytes = cv2.imencode('.jpg', results[0]['bytes'])[1].tobytes()
          image = base64.b64encode(image_bytes).decode("utf-8")
          return render_template('uploadresults.html', char = character, img = image)
      else  :
          return 'No Faces Found!'



if __name__ == "__main__":
    app.run(host='0.0.0.0',port = 5000,debug=True, threaded=True)

