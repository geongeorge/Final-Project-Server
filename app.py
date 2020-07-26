from flask import Flask, escape, request
import werkzeug
import cv2
from utils.model_tool import loadModel,runModelOn,getResultFromPrediction
from utils.image_tool import segment_image

# Initialize by loading the tf model
model = loadModel()

app = Flask(__name__)

UPLOADS = 'uploads/'

@app.route('/')
def hello():
    return 'Hello, world!'

@app.route('/image', methods = ['POST'])
def saveImage():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    print("Saved as "+filename)
    imagefile.save(UPLOADS + filename)
    result = _processImage(filename)
    prediction = { 'status': 'ok', 'result': []}
    for res in result:
        pred = getResultFromPrediction(res)
        prediction['result'].append(pred)

    return prediction

def _processImage(filename):
    segmented = segment_image(filename)
    # Result
    result = []
    for img in segmented:
        res = runModelOn(model, img)
        result.append(res)
    return result