#-*- coding:utf-8 -*- 
import os,random
from flask import render_template,request
from app import app
from werkzeug import secure_filename

from app.midware.yyb_ad_information import get_app_ad_info, rand_ad_info

@app.route('/', methods = ['GET'])
@app.route('/index/', methods = ['GET'])
@app.route('/yybadinfo/random/', methods = ['GET'])
@app.route('/yybadinfo/', methods = ['GET', 'POST'])
def yyb_ad_information():
	render_dict={}
	has_appid = False
	if request.method == 'POST':
		try:
                        appid = request.form.get("appid")
		except:
			errors = []
                        errors.append('no app id')
			render_dict['errors'] = errors
			return render_template('yyb_ad_information.html', **render_dict)
                if len(appid) == 0:
                        errors = []
                        errors.append('no app id')
			render_dict['errors'] = errors
			return render_template('yyb_ad_information.html', **render_dict)
                has_appid = True
	elif request.url.find('/yybadinfo/random') != -1:
                ad = rand_ad_info()
                render_dict['app_ad_analysis_result'] = ad
                render_dict['keynum'] = len(ad['keyword'])
	if has_appid is True:
                ad = get_app_ad_info(appid)
                if len(ad) == 0:
                    errors = []
                    errors.append('no this app')
                    render_dict['errors'] = errors
                    return render_template('yyb_ad_information.html', **render_dict)
                render_dict['app_ad_analysis_result'] = ad
                render_dict['keynum'] = len(ad['keyword'])
	return render_template('yyb_ad_information.html', **render_dict)


