import os

name = 'matmul_M65536N4096K1024_1000'
num = 790

json_dir = name + '_json'
cu_dir = name + '_cu'
if not os.path.exists(cu_dir):
    os.mkdir(cu_dir)
gridblk_dir = name + '_gridblk'
if not os.path.exists(gridblk_dir):
    os.mkdir(gridblk_dir)

f=open("run_gen_cu.sh", 'w')
for i in range(num):
    json_path = os.path.join(json_dir, 'matmul_' + str(i) + '.json')
    cu_path = os.path.join(cu_dir, 'matmul_' + str(i) + '.cu')
    gridblk_path = os.path.join(gridblk_dir, 'matmul_' + str(i))
    f.write('python matmul_rerun.py --config1 ' + json_path + ' --config2 ' + cu_path + ' &> ' + gridblk_path + '\n')
f.close()