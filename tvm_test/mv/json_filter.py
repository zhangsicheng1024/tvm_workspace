import json
import os

name = 'mv_N32768K8192_ori_1000'
threshold = 0.01

file_in_path = name + '.json'
file_out_path = name + '_out.json'
file_in = open(file_in_path, 'r')
file_out = open(file_out_path, 'w')

while True:
    line = file_in.readline()
    if not line: break
    injson = json.loads(line)
    if float(injson['r'][0][0]) < threshold:
        file_out.write(line)

file_in.close()
file_out.close()
