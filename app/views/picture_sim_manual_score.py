#-*- coding:utf-8 -*-
import os
from flask import render_template,request, make_response
from app import app
from werkzeug import secure_filename
from app.midware.picture_sim_manual_score import generate_picture_pair
import lib


@app.route('/pic_sim_man_score/', methods=['GET', 'POST'])
def pic_sim_man_score():
	render_dict={}
	render_dict['auth'] = lib.auth_by_ip(request.remote_addr)
	if request.cookies.get('pic_comp_index'):
		pic_comp_index = int(request.cookies.get('pic_comp_index'))
	else :
		pic_comp_index = 0

	print pic_comp_index
	pic_group_num = app.config['PIC_SIM_MANNUAL_PIC_GROUP_NUM']
	pic_pair_group,pic_comp_index = generate_picture_pair(pic_group_num, pic_comp_index)
	for i in range(len(pic_pair_group)):
		pic_pair_group[i].append('radio'+str(i))
	
	render_dict['pic_pair_group'] = pic_pair_group
	render_dict['pic_group_num'] = pic_group_num
	
	if request.method == 'POST':
		for i in range(pic_group_num):
			score = request.form.get('radio'+str(i))
			if score is None:
				continue
			pic1 = request.cookies.get('.'.join(['score_pic', str(i), '1']))
			pic2 = request.cookies.get('.'.join(['score_pic', str(i), '2']))
			
			cmd = 'echo "'+' '.join([pic1, pic2, str(score)])+'" >> '+ app.config['PIC_SIM_MANUAL_RESULT']
			os.system(cmd)

	resp = make_response(render_template('picture_sim_manual_score.html', **render_dict))

	resp.set_cookie('pic_comp_index', str(pic_comp_index))
	for i in range(pic_group_num):
		pic1 = '.'.join(['score_pic', str(i), '1'])
		pic2 = '.'.join(['score_pic', str(i), '2'])
		resp.set_cookie(pic1, pic_pair_group[i][0])
		resp.set_cookie(pic2, pic_pair_group[i][1])
	return resp
