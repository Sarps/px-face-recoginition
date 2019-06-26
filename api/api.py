from threading import Thread
from PIL import Image
from utils.base_64 import base64_decode_image , base64_encode_image
from train_model import train_face
from test2 import extract_embed
from test import recognize_person
import numpy as np
import base64
from imutils import paths
import flask
from flask import flash,request,redirect,url_for,session
from werkzeug.utils import secure_filename
# from flask_cors import CORS,cross_origin
import time
import json
import sys
import os

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
 
# initialize constants used for server queuing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

UPLOAD_PATH  = '../dataset/'

# initialize our Flask application, Redis server, and Keras model
app = flask.Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH

model = None
user_images = []
path = ""
name = ""

@app.route("/register", methods=["POST"])
def register():
	target = os.path.join(UPLOAD_PATH,name)
	if not os.path.isdir(target):
		os.mkdir(target)
	file = request.files['file']			
	filename = secure_filename(file.filename)
	destination  = "/".join([target,filename])
	file.save(destination)
	session['uploadFilePath'] = destination
	# extract embeddings for new data
	extract_embed()
	train_face()
    # return the data dictionary as a JSON response
    

@app.route("/predict" , methods=["POST"])
def predict():
	
	recognize_person()





# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	# load the function used to classify input images in a *separate*
	# thread than the one used for main classification
	print("* Starting model service...")
	# t = Thread(target=classify_process, args=())
	# t.daemon = True
	# t.start()

	# start the web server
	print("* Starting web service...")
	app.run()