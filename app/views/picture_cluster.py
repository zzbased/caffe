#-*- coding:utf-8 -*-
import os
from flask import render_template,request
from app import app
from werkzeug import secure_filename
import lib


@app.route('/piccluster/', methods=['GET', 'POST'])
def picture_cluster():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	return render_template('picture_cluster.html', **render_dict)
