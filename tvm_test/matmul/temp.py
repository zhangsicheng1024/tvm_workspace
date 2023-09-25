import os

name = 'matmul_M8192N4096K1024_energy_1000'

list1 = ['cu', 'gridblk', 'json', 'kernel_cu', 'ncu', 'tiling']
list2 = ['power_sample', 'power_data', 'temperature']

if not os.path.exists(name): os.mkdir(name)

for end in list1:
    os.system('mv ' + name + '_' + end + ' ' + name + '/' + end)
    print('mv ' + name + '_' + end + ' ' + name + '/' + end)

for begin in list2:
    print('mv ' + begin + '_' + name + ' ' + name + '/' + begin)
    os.system('mv ' + begin + '_' + name + ' ' + name + '/' + begin)