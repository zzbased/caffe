#-*- coding:utf-8 -*- 
import os,random
from flask import render_template,request, make_response
from app import app
from werkzeug import secure_filename
from app.midware.picture_optim import picture_optim_midware
import lib

@app.route('/picoptim/test1/', methods = ['GET'])
@app.route('/picoptim/test2/', methods = ['GET'])
@app.route('/picoptim/', methods = ['GET', 'POST'])
def picture_optim():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	has_image_file = False
	if request.method == 'POST':
		try:
			f = request.files['pic']
			assert len(f.filename) > 0
		except:
			errors = []
			errors.append('上传文件错误')
			render_dict['errors'] = errors
			return render_template('picture_optim.html', **render_dict)
		filename = str(random.random())
		media_name = os.path.join(app.config['UPLOAD_DIR'],filename)
		site_media_name = os.path.join('/upload', filename)
		f.save(media_name)
		has_image_file = True
	elif request.url.find('/picoptim/test') != -1:
		if request.url.find('test1') != -1:
			filename = 'demoimg/optim1.jpg'
		elif request.url.find('test2') != -1:
			filename = 'demoimg/optim2.jpg'

		media_name = os.path.join(app.config['MEDIA_ROOT'],filename)
		site_media_name = os.path.join('/site_media', filename)
		has_image_file = True
		
	if has_image_file:
		render_dict['optim_method_select'] = True
		render_dict['ori_image'] = site_media_name
	
	resp = make_response(render_template('picture_optim.html', **render_dict))
	if has_image_file:
		resp.set_cookie('optim_image', site_media_name)
	return resp


@app.route('/picoptim/choosemethod/', methods = ['POST'])
def picture_optim_choosemethod():
	render_dict = {}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	site_media_name = request.cookies.get('optim_image')

	if site_media_name.find('site_media') != -1:
		media_name = site_media_name.replace('/site_media', app.config['MEDIA_ROOT'])
	else:
		media_name = site_media_name.replace('/upload', app.config['UPLOAD_DIR'])

	render_dict['ori_image'] = site_media_name
	render_dict['optim_method_select'] = True

	if request.form.get('luma_abs'):
		luma_abs_para = request.form.get('luma_abs')
	else :
		luma_abs_para = 5.0

	if request.form.get('contra'):
		contra_para = request.form.get('contra')
	else :
		contra_para = 5.0
		
	pic_optim_result = []
	for optim_method_choose in app.config['PIC_OPTIM_METHOD']:
		para = 5.0
		if optim_method_choose == 'luma_abs':
			para = luma_abs_para
		elif optim_method_choose == 'contra':
			para = contra_para
		result = picture_optim_midware(media_name, optim_method_choose, float(para))
		if result is not None:
			pic_optim_result.append(result)
	render_dict['pic_optim_result'] = pic_optim_result
	render_dict['optim_method_num'] = len(pic_optim_result)
	
	resp = make_response(render_template('picture_optim.html', **render_dict))

	return resp
