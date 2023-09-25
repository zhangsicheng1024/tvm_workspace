import os

name = 'mv_N32768K8192_ori_1000'

json_dir = name + '_json'
if not os.path.exists(json_dir):
    os.mkdir(json_dir)

file_in_path = name + '_out.json'
file_in = open(file_in_path)
lines = file_in.read()
lines = lines.split('\n')

for i, line in enumerate(lines):
    file_out_path = os.path.join(json_dir, 'mv_' + str(i) + '.json')
    file_out = open(file_out_path, 'w')
    file_out.write(lines[i])
    file_out.close()
file_in.close()