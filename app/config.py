import os

# Server address
host='10.137.15.78'  #'10.135.14.15'
port=33359  #5571
kBufferLength = 100000
ServerAddress = (host,port)

# upload image dir
upload_dir="/data/vincentyao/appdemo/app/uploads/"

# old config
PROJECT_DIR = os.path.dirname(__file__)
TMPFILE_DIR = os.path.join(PROJECT_DIR, 'tmpfile')
UPLOAD_DIR = os.path.join(PROJECT_DIR, 'upload')
MEDIA_ROOT = os.path.join(PROJECT_DIR, 'media')
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')
PULLFILE_DIR = os.path.join(PROJECT_DIR, 'pullfile')
RESULT_DIR = os.path.join(PROJECT_DIR, 'result_data')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
ALL_APP_AD_FILE = os.path.join(DATA_DIR, 'ad_app_info')


PIC_OPTIM_METHOD = ['stretchy', 'adjmethod', 'luma_abs', 'contra' ]
PIC_DISSIM_FEEDBACK_NUM = 100
PIC_SIM_MANUAL_RESULT = os.path.join(PROJECT_DIR, 'result_data/sim_manual.txt')
PIC_OPTIM_EVALUATE = os.path.join(PROJECT_DIR, 'result_data/optim_evaluate_manual.txt')