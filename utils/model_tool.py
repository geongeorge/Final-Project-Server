# Provides util functions for:
# loading the model 
# running the model

import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np

# Import the model constants
from utils.constants import PROJECT_HOME,MODEL_FILE,MODEL_DICT

# Loads the model 
def loadModel():
    return keras.models.load_model(MODEL_FILE)

# Reshape image for the model
def reshapeImage(im):
    img = im.reshape(1,32,32,1).astype('float32')
    img /= 255
    return img

# Run model on image
def runModelOn(model,img):
    return np.argmax(model.predict(img))

# Shows the result from the prediciton number
def getResultFromPrediction(result):
    db=pd.read_csv(MODEL_DICT, sep=',',header=None)
    return db[2][result]


