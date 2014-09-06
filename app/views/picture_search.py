#-*- coding:utf-8 -*-
import os
from flask import render_template,request
from app import app
from werkzeug import secure_filename
import lib



@app.route('/picsearch/test2/', methods=['GET'])
@app.route('/picsearch/test1/', methods=['GET'])
@app.route('/picsearch/', methods=['GET', 'POST'])
def picture_search():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	return render_template('picture_search.html', **render_dict)
