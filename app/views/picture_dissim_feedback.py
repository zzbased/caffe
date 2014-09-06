#-*- coding:utf-8 -*-
import os,random,time
from flask import render_template,request, make_response
from app import app
from werkzeug import secure_filename
from app.midware.picture_dissim_feedback import generate_picture_pair
import lib


@app.route('/pic_dissim_feedback/<int:file_id>/', methods=['GET', 'POST'])
def pic_dissim_feedback_process(file_id):
	render_dict={}
	file_list = ['lost_0.7_1', 'lost_0.7_2', 'lost_0.9_1', 'lost_0.9_2', 'new_0.7_1', 'new_0.7_2', 'new_0.9_1', 'new_0.9_2']
	if file_id not in range(len(file_list)):
		return render_template('picture_dissim_feedback.html')
	
	render_dict['this_url'] = '/pic_dissim_feedback/'+str(file_id)+'/'
	dissim_list = os.path.join(app.config['PULLFILE_DIR'], file_list[file_id])

	if request.method == 'GET':
		pic_dissim_index = 0
		render_dict['has_dissim_list'] = True
		total_line = len(open(dissim_list).readlines())
		cur_line = 1
		now_time = time.localtime()
		time_info = '_'.join(map(str, [now_time.tm_mon, now_time.tm_mday, now_time.tm_hour, now_time.tm_min, now_time.tm_sec]))
		result_file_name = os.path.join(app.config['TMPFILE_DIR'], os.path.split(dissim_list)[-1]+'_'+time_info+'_result')
	else:
		render_dict['has_dissim_list'] = True
		pic_dissim_index = int(request.cookies.get('pic_dissim_index'))		
		pic_group_num = int(request.cookies.get('pic_group_num'))
		result_file_name = request.cookies.get('result_file_name')
		cur_line = int(request.cookies.get('cur_line'))
		total_line = int(request.cookies.get('total_line'))
		result_file = open(result_file_name, 'a')
		dissim_num = 0
		rate_file = open(os.path.join(app.config['RESULT_DIR'], 'dis_sim_rate_result'), 'a')
		for i in range(pic_group_num):
			score = request.form.get('radio'+str(i))
			if score is None:
				continue
			pic1 = request.cookies.get('.'.join(['dissim_pic', str(i), '1']))
			pic2 = request.cookies.get('.'.join(['dissim_pic', str(i), '2']))
			if score == "1":
				dissim_num += 1
				rate_file.write(pic1 + ' ' + pic2 +' '+ '0\n')
		rate_file.close()
		result_file.write(' '.join([str(cur_line), str(pic_group_num), str(dissim_num)]) + '\n')
		result_file.close()
		cur_line += 1
		
	pic_group_num = app.config['PIC_DISSIM_FEEDBACK_NUM']
	pic_pair_group,pic_dissim_index = generate_picture_pair(dissim_list, pic_group_num, pic_dissim_index, result_file_name)

	if pic_dissim_index == -2:
		render_dict['has_dissim_list'] = False
		render_dict['total']=pic_pair_group[0]
		render_dict['dissim']=pic_pair_group[1]
		render_dict['dissim_rate']=pic_pair_group[2]
		result_file_name = request.cookies.get('result_file_name')
		result_file = open(result_file_name, 'a')
		result_file.write(str(pic_pair_group[0])+' '+str(pic_pair_group[1])+' ' + str(pic_pair_group[2])+'\n')
		result_file.close()
		return render_template('picture_dissim_feedback_process.html', **render_dict)
			
	for i in range(len(pic_pair_group)):
		pic_pair_group[i].append('radio'+str(i))

	render_dict['pic_pair_group'] = pic_pair_group
	render_dict['cur_line'] = cur_line
	remainder = total_line % app.config['PIC_DISSIM_FEEDBACK_NUM']
	render_dict['total_line'] = total_line/app.config['PIC_DISSIM_FEEDBACK_NUM'] + (remainder!=0)
	resp = make_response(render_template('picture_dissim_feedback_process.html', **render_dict))
	resp.set_cookie('pic_dissim_index', str(pic_dissim_index))
	resp.set_cookie('result_file_name', result_file_name)
	resp.set_cookie('pic_group_num', str(len(pic_pair_group)))
	resp.set_cookie('cur_line', str(cur_line))
	resp.set_cookie('total_line', str(total_line))
	for i in range(len(pic_pair_group)):
		pic1 = '.'.join(['dissim_pic', str(i), '1'])
		pic2 = '.'.join(['dissim_pic', str(i), '2'])
		resp.set_cookie(pic1, pic_pair_group[i][0])
		resp.set_cookie(pic2, pic_pair_group[i][1])
	return resp



@app.route('/pic_dissim_feedback/', methods=['GET'])
def pic_dissim_feedback():
	return render_template('picture_dissim_feedback.html')
