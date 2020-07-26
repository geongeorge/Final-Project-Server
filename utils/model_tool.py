# Provides util functions for:
# loading the model 
# running the model

import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np
# Resize image
from utils.resize_image import resizeImage

# Import the model constants
from utils.constants import PROJECT_HOME,MODEL_FILE,MODEL_DICT

# Loads the model 
def loadModel():
    return keras.models.load_model(MODEL_FILE)

# Reshape image for the model
def _reshapeImage(im):
    status,raw_img = resizeImage(im,32)
    img = raw_img.reshape(1,32,32,1).astype('float32')
    img /= 255
    return img

# Run model on image
def runModelOn(model,img):
    im = _reshapeImage(img)
    return np.argmax(model.predict(im))

# Shows the result from the prediciton number
def getResultFromPrediction(result):
    db=pd.read_csv(MODEL_DICT, sep=',',header=None)
    return db[2][result]


