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