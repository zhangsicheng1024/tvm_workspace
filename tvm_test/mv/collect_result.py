import csv
import json
import os

name = 'mv_N32768K8192_ori_1000_2'
num = 1000

f_csv=open('results_' + name + '.csv', 'w')
csv_writer = csv.writer(f_csv)

power_max = 0
power_min = 10000
# list = ['dram_l', 'dram_s', 'l2_l', 'l2_s', 'shared_l', 'shared_s', 'sm_efficiency(%)', 'flop_efficiency', 'lat_tvm(ms)', 'lat_nvprof', 'energy(mJ)', 'power_peak(W)', 'power_avg', 'temp_avg(C)', 'index', 'tp_dram_l(GB/s)', 'tp_dram_s', 'tp_l2_l', 'tp_l2_s', 'tp_shared_l', 'tp_shared_s']
list = ['dram_l', 'dram_s', 'l2_l', 'l2_s', 'shared_l', 'shared_s', 'sm_efficiency(%)', 'flop_efficiency', 'lat_tvm(ms)', 'lat_nvprof', 'energy(mJ)', 'power_peak(W)', 'power_avg', 'temp_avg(C)', 'index']
csv_writer.writerow(list)

run_list = range(0, num)
# run_list = [64,640,768,641,642,704,832,643,833,948,644,949,705,128,896,950,951,952,769,953,954,770,771,955,956,645,897,256,257,957,834,706,835,958,959,960,320,707,772,836,837,961,962,963,838,321,839,898,773,964,708,899,448,774,900,322,384,775,965,709,323,324,710,966,449,967,325,776,968,450,840,841,326,258,777,327,969,385,778,259,779,260,970,842,711,971,780,386,512,387,843,712,261,844,781,713,451,845,846,388,]
for i in run_list:
    file_json = open(os.path.join(name + '_json', 'mv_' + str(i) + '.json'), 'r')
    line = file_json.readline()
    injson = json.loads(line)
    lat_tvm = injson['r'][0][0] # s
    if lat_tvm == 1e10: continue
    lat_tvm = round(lat_tvm * 1000, 4) # ms

    file_power = open(os.path.join('power_data_' + name, 'power_data_' + str(i)), 'r')
    energy = round(float(file_power.readline()), 3)
    power_avg = round(float(file_power.readline()), 2)
    power_peak = round(float(file_power.readline()), 2)

    file_temperature = open(os.path.join('temperature_' + name, 'temperature_' + str(i)), 'r')
    temp_list = file_temperature.readlines()
    temp_list = [int(temp.strip()) for temp in temp_list]
    temp_avg = round(sum(temp_list) / len(temp_list), 3)

    file_nvprof_latency = open(os.path.join(name + '_ncu', 'mv_' + str(i) + '_latency'), 'r')
    while True:
        text = file_nvprof_latency.readline()
        if text == "": break
        # print(text)
        text = text.split()
        if len(text) == 0: continue
        if text[-1].endswith("kernel0"):
            if text[-6][-2:]=='us': base = 1e-3
            elif text[-6][-2:]=='ms': base = 1
            else: print('error!')
            lat_nvprof = round(float(text[-6][:-2])*base, 5) # ms
            break

    file_nvprof_metrics = open(os.path.join(name + '_ncu', 'mv_' + str(i) + '_metrics'), 'r')
    while True:
        text = file_nvprof_metrics.readline()
        if text == "": break
        # print(text)
        text = text.split()
        if len(text) == 0: continue
        if text[1] == "dram_read_transactions":
            dram_l = int(text[-1])
        elif text[1] == "dram_write_transactions":
            dram_s = int(text[-1])
        elif text[1] == "l2_read_transactions":
            l2_l = int(text[-1])
        elif text[1] == "l2_write_transactions":
            l2_s = int(text[-1])
        elif text[1] == "shared_load_transactions":
            shared_l = int(text[-1])
        elif text[1] == "shared_store_transactions":
            shared_s = int(text[-1])
        elif text[1] == "sm_efficiency":
            sm_efficiency = float(text[-1][:-1])
        elif text[1] == "flop_sp_efficiency":
            flop_efficiency = float(text[-1][:-1])
        # elif text[1] == "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum":
        #     fma_flops = int(text[-1])
        # elif text[1] == "dram_read_throughput":
        #     throughput_dram_l = float(text[-1][:-4])
        # elif text[1] == "dram_write_throughput":
        #     throughput_dram_s = float(text[-1][:-4])
        # elif text[1] == "l2_read_throughput":
        #     throughput_l2_l = float(text[-1][:-4])
        # elif text[1] == "l2_write_throughput":
        #     throughput_l2_s = float(text[-1][:-4])
        # elif text[1] == "shared_load_throughput":
        #     throughput_shared_l = float(text[-1][:-4])
        # elif text[1] == "shared_store_throughput":
        #     throughput_shared_s = float(text[-1][:-4])
    # l2_l = l2_l + l2_base1 + l2_base2
    # l2_s = l2_s + l2_base1 + l2_base2
    # throughput_l2_l = throughput_l2_l + throughput_l2_base1 + throughput_l2_base2
    # throughput_l2_s = throughput_l2_s + throughput_l2_base1 + throughput_l2_base2


    list = [dram_l, dram_s, l2_l, l2_s, shared_l, shared_s, sm_efficiency, flop_efficiency, lat_tvm, lat_nvprof, energy, power_peak, power_avg, temp_avg, i]
    # print(list)
    file_json.close()
    file_power.close()
    file_temperature.close()
    file_nvprof_latency.close()
    file_nvprof_metrics.close()
    csv_writer.writerow(list)
f_csv.close()

# print("power_max: ", power_max, id_max)
# print("power_min: ", power_min, id_min)