#-*- coding:utf-8 -*- 
import os
from flask import render_template,request, make_response
from app import app
from werkzeug import secure_filename
from app.midware.picture_optim import picture_optim_midware, generate_optim_image
import lib


@app.route('/picoptimfeedback/', methods = ['GET', 'POST'])
def picture_optim_feedback():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	try:
		site_media_name = request.cookies.get('optim_image_feedback')
		use_method = request.cookies.get('use_optim_method').split(' ')
	except:
		pass

	if request.cookies.get('optim_index'):
		optim_index = int(request.cookies.get('optim_index'))
	else :
		optim_index = 0
		
	if request.form.get('score0'):
		content = site_media_name
		for i in range(len(use_method)):
			score = request.form.get('score'+str(i))
			if score != "undefined":
				content += " " + use_method[i] + ":" + score
			else :
				content += " " + use_method[i] + ":-1"
		cmd = 'echo "'+content+'" >> '+ app.config['PIC_OPTIM_EVALUATE']
		os.system(cmd)
		return render_template('picture_optim.html', **render_dict)


	filename,optim_index = generate_optim_image(optim_index)
	media_name = os.path.join(app.config['MEDIA_ROOT'], filename)
	site_media_name = os.path.join('/site_media', filename)

	use_method = []
	pic_optim_result = []
	for optim_method_choose in app.config['PIC_OPTIM_METHOD']:
		para = 5.0
		if optim_method_choose == 'luma_abs':
			para = 5.0
		elif optim_method_choose == 'contra':
			para = 5.0
		result = picture_optim_midware(media_name, optim_method_choose, float(para))
		if result is not None:
			pic_optim_result.append(result)
			use_method.append(optim_method_choose)
			
	render_dict['ori_image'] = site_media_name
	render_dict['pic_optim_result'] = pic_optim_result
	render_dict['optim_method_num'] = len(pic_optim_result)
	
	resp = make_response(render_template('picture_optim_feedback.html', **render_dict))
	resp.set_cookie('use_optim_method', ' '.join(use_method))
	resp.set_cookie('optim_image_feedback', site_media_name)
	resp.set_cookie('optim_index', str(optim_index))

	return resp
