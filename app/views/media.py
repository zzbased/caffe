import os, random
from flask import render_template,request,send_from_directory
from app import app


@app.route('/site_media/<path:filename>/', methods=['GET'])
def site_media(filename):
	return send_from_directory(app.config['MEDIA_ROOT'], filename)


@app.route('/upload/<filename>/', methods=['GET'])
def upload_dir(filename):
	return send_from_directory(app.config['UPLOAD_DIR'], filename)



@app.route('/tmpfile/<filename>/', methods=['GET'])
def tmp_file(filename):
	return send_from_directory(app.config['TMPFILE_DIR'], filename)
