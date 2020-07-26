# script to test the models working

import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np

from utils.resize_image import openAndResize

# Constants 

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = PROJECT_HOME+"/model/ora.h5"
MODEL_DICT = PROJECT_HOME+'/model/dict_output.csv'

INPUT_IMAGE = "sample/11.png"

# Loading Model 
model = keras.models.load_model(MODEL_FILE)

# Read the csv file in pandas
db=pd.read_csv(MODEL_DICT, sep=',',header=None)

# Load sample Image
status,im = openAndResize(INPUT_IMAGE, 32)

img = im.reshape(1,32,32,1).astype('float32')
img /= 255

result = np.argmax(model.predict(img))

print(db[2][result])


#resize img
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
