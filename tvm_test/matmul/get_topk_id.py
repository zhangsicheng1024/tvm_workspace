import numpy as np

name = 'matmul_M1024N4096K1024_energy_pc150+150_1000'

path = 'energy_record_' + name + '.txt'
path = 'energy_record3.txt'
k = 200

energy_list = np.array([])
with open(path) as file:
    while True:
        line = file.readline()
        if not line: break
        # energy_list.append(float(line))
        energy_list = np.append(energy_list, float(line))

sorted_id = np.argsort(energy_list)

topk = sorted_id[:k]



print('topk index')
print('[', end='')
for i in topk:
    print(str(i) + ',', end='')
print(']')

print('topk value')
print(energy_list[topk])



# energy_compare = []
# min2 = 10000

# for i in topk:
#     path2 = 'power_data_' + name + '/power_data_' + str(i) + '.txt'
#     with open(path2) as file:
#         line = file.readline()
#     energy_compare.append((i, energy_list[i], float(line)))
#     min2 = min(min2, float(line))

# for i in energy_compare:
#     print(i)

# print('min1', energy_list[sorted_id[0]])
# print('min2', min2)