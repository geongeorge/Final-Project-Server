from flask import Flask, escape, request
from flask_cors import CORS,cross_origin
import werkzeug
import shortuuid
import cv2
from utils.model_tool import loadModel,runModelOn,getResultFromPrediction
from utils.image_tool import segment_image

# Initialize by loading the tf model
model = loadModel()

app = Flask(__name__)
cors = CORS(app)

UPLOADS = 'uploads/'

@app.route('/')
@cross_origin()
def hello():
    return 'Hello, world!'

@app.route('/image', methods = ['POST'])
def saveImage():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    print("Saved as "+filename)
    extension = filename.split(".")[-1]
    uuid = shortuuid.uuid()
    filename = uuid+'.'+extension
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