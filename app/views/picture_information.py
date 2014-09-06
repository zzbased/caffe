#-*- coding:utf-8 -*- 
import os,random
from flask import render_template,request
from app import app
from werkzeug import secure_filename
from app.midware.picture_information import picture_information_midware
import lib

@app.route('/', methods = ['GET'])
@app.route('/index/', methods = ['GET'])
@app.route('/picinfo/test1/', methods = ['GET'])
@app.route('/picinfo/test2/', methods = ['GET'])
@app.route('/picinfo/', methods = ['GET', 'POST'])
def picture_info():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	checkbox_dict = []
	for i in range(len(app.config['PICINFO'])):
		checkbox_dict.append((app.config['PICINFO'][i],app.config['PICINFO_DES'][i], "1"))
	render_dict['checkbox_dict'] = checkbox_dict

	has_image_file = False
	if request.method == 'POST':
		try:
			f = request.files['pic']
			assert len(f.filename) > 0
		except:
			errors = []
			errors.append('上传文件错误')
			render_dict['errors'] = errors
			return render_template('picture_information.html', **render_dict)
		filename = str(random.random())
		media_name = os.path.join(app.config['UPLOAD_DIR'],filename)
		site_media_name = os.path.join('/upload', filename)
		f.save(media_name)
		pic_info_list = []
		for i in range(len(app.config['PICINFO'])):
			if request.form.get(app.config['PICINFO'][i]):
				pic_info_list.append(app.config['PICINFO'][i])
		has_image_file = True
	elif request.url.find('/picinfo/test') != -1:
		if request.url.find('test1') != -1:
			filename = 'demoimg/info1.jpg'
		elif request.url.find('test2') != -1:
			filename = 'demoimg/info2.jpg'
		media_name = os.path.join(app.config['MEDIA_ROOT'],filename)
		site_media_name = os.path.join('/site_media', filename)
		pic_info_list = []
		for i in range(len(app.config['PICINFO'])):
			pic_info_list.append(app.config['PICINFO'][i])
		has_image_file = True
		
	if has_image_file:
		pic_info_result = picture_information_midware(pic_info_list, media_name)
		render_dict['ori_image'] = site_media_name
		render_dict['pic_info_result'] = pic_info_result
	return render_template('picture_information.html', **render_dict)


