# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np

# Must review this function
def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")
 
	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)
 
	# return the processed image
	return image