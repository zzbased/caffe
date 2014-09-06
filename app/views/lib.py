import os
from app import app




def auth_by_ip(ip):
	f=open(app.config['WHITE_LIST_NAME'])
	for i in f:
		if ip == i.strip():
			return True
	return False

