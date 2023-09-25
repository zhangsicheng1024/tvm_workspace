import numpy as np

path = 'energy_record'
k = 100

energy_list = np.array([])
with open(path) as file:
    while True:
        line = file.readline()
        if not line: break
        # energy_list.append(float(line))
        energy_list = np.append(energy_list, float(line))

sorted_id = np.argsort(energy_list)

topk = sorted_id[:k]

# print('topk index')
# print('[', end='')
# for i in topk:
#     print(str(i) + ',', end='')
# print(']')

# print('topk value')
# print(energy_list[topk])

energy_compare = []

for i in topk:
    path2 = 'power_data_dwconv_N128C128H28W28_energy_1000_2/power_data_' + str(i)
    with open(path2) as file:
        line = file.readline()
    energy_compare.append((i, energy_list[i], float(line)))

for i in energy_compare:
    print(i)