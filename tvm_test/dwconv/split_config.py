import os

name = 'dwconv_N128C128H28W28_energy_500'

json_dir = name + '_json'
if not os.path.exists(json_dir):
    os.mkdir(json_dir)

file_in_path = name + '_out.json'
file_in = open(file_in_path)
lines = file_in.read()
lines = lines.split('\n')

for i, line in enumerate(lines):
    file_out_path = os.path.join(json_dir, 'dwconv_' + str(i) + '.json')
    file_out = open(file_out_path, 'w')
    file_out.write(lines[i])
    file_out.close()
file_in.close()