from flask import Flask, escape, request
import werkzeug

app = Flask(__name__)

UPLOADS = 'uploads/'

@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

@app.route('/image', methods = ['POST'])
def processImage():
    imagefile = request.files['image']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(UPLOADS + filename)
    return 'Success !'