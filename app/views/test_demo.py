#-*- coding:utf-8 -*-
import os,random
from flask import render_template,request
from app import app
from werkzeug import secure_filename
import lib



@app.route('/testdemo/', methods=['GET', 'POST'])
def test_demo():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)

	if request.method == 'POST':
		try:
			f = request.files['data']
			assert len(f.filename) > 0
		except:
			errors = []
			errors.append('\344\270\212\344\274\240\346\226\207\344\273\266\351\224\231\350\257\257')
			render_dict['errors'] = errors
			return render_template('test_demo.html', **render_dict)
		filename = str(random.random())
		media_name = os.path.join(app.config['UPLOAD_DIR'],filename)
		f.save(media_name)
		render_dict['test_demo_result'] = 20
	return render_template('test_demo.html', **render_dict)
