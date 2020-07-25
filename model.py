import tensorflow as tf
from tensorflow import keras
import cv2
import pandas as pd
import os
import numpy as np

# Constants 

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = PROJECT_HOME+"/model/ora.h5"
MODEL_DICT = PROJECT_HOME+'/model/dict_output.csv'

INPUT_IMAGE = "sample/11.png"

# Opens and resizes the image using opencv
def openAndResize(path,size):
    try:
        im = cv2.imread(path, 0)
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return 1,new_im
    except:
        print(sys.exc_info()[0])
        print("Error Image : ",path)
        return -1,[] #error

# Loading Model 
model = keras.models.load_model(MODEL_FILE)

# Read the csv file in pandas
db=pd.read_csv(MODEL_DICT, sep=',',header=None)

# Load sample Image
status,im = openAndResize(INPUT_IMAGE, 32)

img = im.reshape(1,32,32,1).astype('float32')
img /= 255

result = np.argmax(model.predict(img))
    # alphabet = next(key for key, value in labels_values.items() if value == result) #reverse key lookup
# original = db[result][2]
# print("Prediction"+original)
print(db[2][result])