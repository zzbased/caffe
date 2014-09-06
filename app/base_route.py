import os,sys
import socket
import time
import struct
reload(sys)
sys.setdefaultencoding('UTF-8')
# add python lib include dir
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'pythonlib'))
sys.path.append(os.path.join(PROJECT_DIR, 'proto'))
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Initialize the Flask application
app = Flask(__name__)
# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/image_classify/')
def classify():
    return render_template('image_classify.html')

@app.route('/image_similarity/')
def similarity():
    return render_template('image_similarity.html')

@app.route('/image_search/')
def search():
    return render_template('image_search.html')

@app.route('/image_semantic/')
def semantic():
    return render_template('image_semantic.html')

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/site_media/<filename>')
def site_media(filename):
    return send_from_directory("site_media/", filename)

@app.route('/gdt_media/<filename>')
def gdt_media(filename):
    return send_from_directory("gdt_media/", filename)