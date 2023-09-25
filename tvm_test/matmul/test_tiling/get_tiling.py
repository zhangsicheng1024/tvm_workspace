import re
import argparse
import json
import os

parser = argparse.ArgumentParser(description='mv kernel names')
parser.add_argument('--json')
args_top = parser.parse_args()

M = 512
N = 4096
K = 1024

id = 6
path_json_in = 'matmul_' + str(id) + '.json'
if args_top.json: path_json_in = args_top.json
path_rerun = '_tmp/matmul_rerun.py'
path_json = '_tmp/tmp.json'
path_cu = '_tmp/tmp.cu'
path_gridblk = '_tmp/tmp_gridblk.txt'

# json preprocess
file_json_in = open(path_json_in, 'r')
file_json = open(path_json, 'w')
line = file_json_in.readline().strip()
injson = json.loads(line)
unroll_factor = int(injson['i'][-1][-1][-1][-1].split('$')[-1])
injson['i'][-1][-1][-1][-1] = 'auto_unroll_max_step$0'
file_json.write(json.dumps(injson))
file_json_in.close()
file_json.close()

# generate cuda and gridblk
command_gridblk_cu = 'python ' + path_rerun + ' --config1 ' + path_json + ' --config2 ' + path_cu + ' > ' + path_gridblk
os.system(command_gridblk_cu)

# get cuda and gridblk
file_gridblk = open(path_gridblk, 'r')
block_num = int(file_gridblk.readline().split()[0])
thread_num = int(file_gridblk.readline().split()[0])
file_gridblk.close()

file_cu = open(path_cu, 'r')
lines = file_cu.readlines()
file_cu.close()

vthreads_num = 0
vthreads_offset = 0
vthreads_data_list = []
vthreads_kernel_list = []
index = 0
for line in lines:
    line = line.strip()
    m_local = re.match('float T_matmul_NN_local\[([0-9]+)\];', line)
    m_data = re.match('__shared__ float data_shared\[([0-9]+)\];', line)
    m_kernel = re.match('__shared__ float kernel_shared\[([0-9]+)\];', line)
    m_k_out = re.match('for \(int k_outer_outer = 0; k_outer_outer < ([0-9]+); \+\+k_outer_outer\) \{', line)
    m_vthread = re.match('T_matmul_NN_local\[(.+)\] = \(T_matmul_NN_local\[(.+)\] \+ \(data_shared\[(.+)\] \* kernel_shared\[(.+)\]\)\);', line)
    if(m_local):
        local_size = int(m_local.group(1))
    elif(m_data):
        data_shared_size = int(m_data.group(1))
    elif(m_kernel):
        kernel_shared_size = int(m_kernel.group(1))
    elif(m_k_out):
        K_out = int(m_k_out.group(1))
    elif(m_vthread):
        vthreads_num += 1
        if vthreads_offset == 0: vthreads_offset = index
        vthreads_data_list.append(m_vthread.group(3))
        vthreads_kernel_list.append(m_vthread.group(4))
    index += 1

for_list = []
index = vthreads_offset - 1
while index >= 0:
    line = lines[index].strip()
    m_for = re.match('for \(int (.+) = 0; (.+) < ([0-9]+); \+\+(.+)\) \{', line)
    if not m_for: break
    for_list.append([m_for.group(1), int(m_for.group(3))])
    index -= 1
for_list.reverse()

# block level
K_in = int(K / K_out)
M_block = int(data_shared_size / K_in)
N_block = int(kernel_shared_size / K_in)
i_block = int(M / M_block)
j_block = int(N / N_block)
assert i_block * j_block == block_num

i_local_list = []
j_local_list = []
k_local_list = []

# local level
for f in range(len(for_list)):
    if for_list[f][0][0] == 'i':
        new_name = 'i_local_' + str(len(i_local_list))
        for_list[f][0] = new_name
        i_local_list.append(int(for_list[f][1]))
    elif for_list[f][0][0] == 'j':
        new_name = 'j_local_' + str(len(j_local_list))
        for_list[f][0] = new_name
        j_local_list.append(int(for_list[f][1]))
    elif for_list[f][0][0] == 'k':
        new_name = 'k_local_' + str(len(k_local_list))
        for_list[f][0] = new_name
        k_local_list.append(int(for_list[f][1]))
    else:
        print('error')

vthread_data_dict = {}
if vthreads_num != 1:
    for vthread_data in vthreads_data_list:
        if not vthread_data_dict.get(vthread_data):
            vthread_data_dict[vthread_data] = 1
    vthread_i = len(vthread_data_dict)
    vthread_j = int(vthreads_num / len(vthread_data_dict))
    i_local_list.insert(0, vthread_i)
    j_local_list.insert(0, vthread_j)
    for_list.append(['vthread_loop_i', vthread_i])   # default: vthread_i outerloop
    for_list.append(['vthread_loop_j', vthread_j])
    if vthread_i != 1 and vthread_j != 1 and vthreads_data_list[0] != vthreads_data_list[1]:
        for_list[-1], for_list[-2] = for_list[-2], for_list[-1] # vthread_j outerloop
else:
    for_list.append(['vthread_loop', 1])

k_local = 1
for k_t in k_local_list: k_local *= k_t
assert k_local == K_in

i_local = 1
for i_t in i_local_list: i_local *= i_t

j_local = 1
for j_t in j_local_list: j_local *= j_t

assert i_local * j_local == local_size

# thread level
i_thread = int(M_block / i_local)
j_thread = int(N_block / j_local)
assert i_thread * j_thread == thread_num

M_thread = i_local
N_thread = j_local

# unroll pos
unroll_lines = 0
unroll_count = 1
for f in reversed(for_list):
    if f[0][:7] == 'vthread' or unroll_count * f[1] <= unroll_factor:
        unroll_lines += 1
        unroll_count *= f[1]
    else: break
unroll_start = len(for_list) - unroll_lines

not_unrolled_loop_size = 1
M_unroll = M_thread
N_unroll = N_thread
K_unroll = K_in
for i in range(unroll_start):
    not_unrolled_loop_size *= for_list[i][1]
    if for_list[i][0][0] == 'i':
        M_unroll = int(M_unroll / for_list[i][1])
    elif for_list[i][0][0] == 'j':
        N_unroll = int(N_unroll / for_list[i][1])
    elif for_list[i][0][0] == 'k':
        K_unroll = int(K_unroll / for_list[i][1])

# output
print('===========compute params===========')
print('M=%d, N=%d, K=%d' % (M, N, K))
print("block_num=%d, thread_num=%d" % (block_num, thread_num))
print('local_size=%d, data_shared_size=%d, kernel_shared_size=%d' % (local_size, data_shared_size, kernel_shared_size))
print()

print('===========block===========')
print('for i_block in', i_block)
print('for j_block in', j_block)
print('for k_out in', K_out)
print('per block: data[%d*%d] x kernel[%d*%d] = ret[%d*%d]' % (M_block, K_in, K_in, N_block, M_block, N_block))
print()

print('===========thread===========')
print('for i_thread in', i_thread)
print('for j_thread in', j_thread)
print('per thread: data[%d*%d] x kernel[%d*%d] = ret[%d*%d]' % (M_thread, K_in, K_in, N_thread, M_thread, N_thread))
print()

print('===========local===========')
for i, f in enumerate(for_list):
    if(i == unroll_start): print('------------------------------unrolled: data[%d*%d] x kernel[%d*%d] = ret[%d*%d] (x%d)' % (M_unroll, K_unroll, K_unroll, N_unroll, M_unroll, N_unroll, not_unrolled_loop_size))
    print('for ' + f[0] + " in " + str(f[1]))
print('local[] = local[] + data_shared[] * kernel_shared[]')
print()

print('==============================Memory transaction==============================')
print('Load (global -> shared)\nglobal read = shared write\n= out loop size * (block data size + block kernel size)')
print('= i_block * j_block * k_out * (MK/(i_block*k_out) + KN/(k_out*j_block))\n= j_block * MK + i_block * KN')
global_read = j_block * M * K + i_block * K * N
print('=', global_read)
print()

print('Compute (shared -> local)')
print('shared read = reg write')
print('= not unrolled loop size * (unrolled data size + unrolled kernel size)')
shared_read_thread = not_unrolled_loop_size * (M_unroll * K_unroll + K_unroll * N_unroll)
print('= %d * ([%d*%d] + [%d*%d]) = %d / thread' % (not_unrolled_loop_size, M_unroll, K_unroll, K_unroll, N_unroll, shared_read_thread))
print('= per_thread * block_num * k_out * thread_num')
shared_read = shared_read_thread * block_num * K_out * thread_num
print('=', shared_read)
print('local write = MNK =', M*N*K)
print()

print('Write back (local -> global)\nlocal_read = global write')
print('= block_num * thread_num * local_size\n= MN')
global_write = M * N
print('=', global_write)
print()