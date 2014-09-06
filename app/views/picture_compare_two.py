#-*- coding:utf-8 -*-
import os,random
from flask import render_template,request,make_response
from app import app
from werkzeug import secure_filename
from app.midware.picture_compare_two import picture_compare_two_normal_midware, picture_compare_two_finger_midware, picture_compare_two_deeplearning_midware
import lib

@app.route('/piccomptwo/test2/', methods=['GET'])
@app.route('/piccomptwo/test1/', methods=['GET'])
@app.route('/piccomptwo/', methods=['GET', 'POST'])
def picture_compare_two():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)	
	comp_checkbox_dict = []
	for i in range(len(app.config['PIC_COMP_METHOD'])):
		choose = "1"
		if app.config['PIC_COMP_METHOD'][i] == "gist_match":
			choose = "0"
		elif app.config['PIC_COMP_METHOD'][i] == 'dl_match':
			choose = "0"
		comp_checkbox_dict.append((app.config['PIC_COMP_METHOD'][i],app.config['PIC_COMP_METHOD_TITLE'][i], choose))
	render_dict['comp_checkbox_dict'] = comp_checkbox_dict

	has_image_file = False
	if request.method == 'POST':
		try:
			f1 = request.files['pic1']
			assert len(f1.filename) > 0
			f2 = request.files['pic2']
			assert len(f2.filename) > 0
		except:
			errors = []
			errors.append('\344\270\212\344\274\240\346\226\207\344\273\266\351\224\231\350\257\257')
			render_dict['errors'] = errors
			return render_template('picture_compare_two.html', **render_dict)
		
		filename1 = str(random.random())
		media_name1 = os.path.join(app.config['UPLOAD_DIR'],filename1)
		site_media_name1 = os.path.join('/upload', filename1)
		f1.save(media_name1)
		
		filename2 = str(random.random())
		media_name2 = os.path.join(app.config['UPLOAD_DIR'],filename2)
		site_media_name2 = os.path.join('/upload', filename2)
		f2.save(media_name2)

		pic_comp_list = []
		for i in range(len(app.config['PIC_COMP_METHOD'])):
			if request.form.get(app.config['PIC_COMP_METHOD'][i]):
				pic_comp_list.append(app.config['PIC_COMP_METHOD'][i])

		if request.form.get(app.config['PIC_COMP_METHOD'][-1]):
			dl_comp = True
		else :
			dl_comp = False
			
		if request.form.get(app.config['PIC_COMP_METHOD'][-1]):
			finger_comp = True
		else :
			finger_comp = False
		
		if request.form.get('dlpara'):
			dllayer = request.form.get('dlpara')
		else :
			dllayer = 24

		has_image_file = True
	elif request.url.find('/piccomptwo/test') != -1:
		if request.url.find('test1') != -1:
			filename1 = 'demoimg/sim1.jpg'
			filename2 = 'demoimg/sim2.jpg'
		elif request.url.find('test2') != -1:
			filename1 = 'demoimg/sim3.jpg'
			filename2 = 'demoimg/sim4.jpg'
		
		media_name1 = os.path.join(app.config['MEDIA_ROOT'],filename1)
		site_media_name1 = os.path.join('/site_media', filename1)

		media_name2 = os.path.join(app.config['MEDIA_ROOT'],filename2)
		site_media_name2 = os.path.join('/site_media', filename2)

		pic_comp_list = []
		for i in range(len(app.config['PIC_COMP_METHOD'])):
			pic_comp_list.append(app.config['PIC_COMP_METHOD'][i])

		dl_comp = False
		finger_comp = True
		dllayer = 24
		has_image_file = True
		
	if has_image_file:
		render_dict['ori_image1'] = site_media_name1
		render_dict['ori_image2'] = site_media_name2
		render_dict['has_image_file'] = has_image_file

		pic_comp_normal_result = picture_compare_two_normal_midware(pic_comp_list, media_name1, media_name2)
		pic_comp_finger_result = None
		if finger_comp:
			pic_comp_finger_result = picture_compare_two_finger_midware(media_name1, media_name2)

		pic_comp_deeplearning_result = None
		if dl_comp:
			pic_comp_deeplearning_result = picture_compare_two_deeplearning_midware(media_name1, media_name2, int(dllayer))
		render_dict['pic_comp_normal_result'] = pic_comp_normal_result
		render_dict['pic_comp_finger_result'] = pic_comp_finger_result
		render_dict['pic_comp_deeplearning_result'] = pic_comp_deeplearning_result
	return render_template('picture_compare_two.html', **render_dict)




