import sys
sys.path.append('/home/shivangi/awd-lstm-lm/cottoncandy')

import cottoncandy as cc
access_key = 'SSE14CR7P0AEZLPC7X0R'
secret_key = 'K0MmeXiXotrGIiTeRwEKizkkhR4qFV8tr8cIXprI'
endpoint_url = 'http://c3-dtn02.corral.tacc.utexas.edu:9002/'
cci = cc.get_interface('lstm-timescales', ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)

import numpy as np
f = open('all_PY','r')
PY_name_list = f.read().splitlines()
for file_name in PY_name_list:
    print(file_name)
    cci.upload_from_file('./'+file_name, 'shivangi/python_backup'+ file_name)
    cci.download_to_file('shivangi/python_backup'+ file_name, 'download'+file_name)
print(len(PY_name_list))
