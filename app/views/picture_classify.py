#-*- coding:utf-8 -*-
import os
from flask import render_template,request
from app import app
from werkzeug import secure_filename
from app.midware.picture_classify import picture_classify_midware
import lib


@app.route('/picclass/test2/', methods = ['GET'])
@app.route('/picclass/test1/', methods = ['GET'])
@app.route('/picclass/', methods = ['GET','POST'])
def picture_classify():
	render_dict = {}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	method_select = []
	for i in range(len(app.config['PIC_CLASS_MODEL'])):
		method_select.append((app.config['PIC_CLASS_MODEL'][i], app.config['PIC_CLASS_MODEL_DES'][i]))
	render_dict['method_select'] = method_select

	has_image_file = False
	if request.method == 'POST':
		try:
			f = request.files['pic']
			assert len(f.filename) > 0
		except:
			errors = []
			errors.append('\344\270\212\344\274\240\346\226\207\344\273\266\351\224\231\350\257\257')
			render_dict['errors'] = errors
			return render_template('picture_classify.html', **render_dict)
		filename = secure_filename(f.filename)
		media_name = os.path.join(app.config['UPLOAD_DIR'],filename)
		site_media_name = os.path.join('/upload', filename)
		f.save(media_name)
		method_choose = request.form.get('model_choose')
		has_image_file = True
	elif request.url.find('/picclass/test') != -1:
		if request.url.find('test1') != -1:
			filename = 'demoimg/class1.jpg'
			method_choose = app.config['PIC_CLASS_MODEL'][0]
		elif request.url.find('test2') != -1:
			filename = 'demoimg/class2.jpg'
			method_choose = app.config['PIC_CLASS_MODEL'][1]
		media_name = os.path.join(app.config['MEDIA_ROOT'],filename)
		site_media_name = os.path.join('/site_media', filename)
		has_image_file = True
		
	if has_image_file:
		pic_classify_result = picture_classify_midware(method_choose, media_name)
		render_dict['ori_image'] = site_media_name
		render_dict['method_choose'] = method_choose
		render_dict['pic_classify_result'] = pic_classify_result
	
	return render_template('picture_classify.html', **render_dict)
