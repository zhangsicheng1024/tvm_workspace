import os

name = 'dwconv_N128C128H28W28_energy_500'
num = 434

json_dir = name + '_json'
cu_dir = name + '_cu'
if not os.path.exists(cu_dir):
    os.mkdir(cu_dir)
gridblk_dir = name + '_gridblk'
if not os.path.exists(gridblk_dir):
    os.mkdir(gridblk_dir)

f=open("run_gen_cu.sh", 'w')
for i in range(num):
    json_path = os.path.join(json_dir, 'dwconv_' + str(i) + '.json')
    cu_path = os.path.join(cu_dir, 'dwconv_' + str(i) + '.cu')
    gridblk_path = os.path.join(gridblk_dir, 'dwconv_' + str(i))
    f.write('python dwconv_rerun.py --config1 ' + json_path + ' --config2 ' + cu_path + ' > ' + gridblk_path + '\n')
f.close()