import csv
import json
import os

name = 'matmul_M1024N4096K1024_ori_pc300+150_1000'
op = name.split('_')[0]

f_csv=open('results_' + name + '.csv', 'w')
csv_writer = csv.writer(f_csv)

power_max = 0
power_min = 10000
# list = ['dram_l', 'dram_s', 'l2_l', 'l2_s', 'shared_l', 'shared_s', 'sm_efficiency(%)', 'flop_efficiency', 'lat_tvm(ms)', 'lat_nvprof', 'energy(mJ)', 'power_peak(W)', 'power_avg', 'temp_avg(C)', 'index', 'tp_dram_l(GB/s)', 'tp_dram_s', 'tp_l2_l', 'tp_l2_s', 'tp_shared_l', 'tp_shared_s']
list = ['dram_l', 'dram_s', 'l2_l', 'l2_s', 'shared_l', 'shared_s', 'sm_efficiency(%)', 'flop_efficiency', 'lat_tvm(ms)', 'lat', 'energy(mJ)', 'power_peak(W)', 'power_avg', 'temp_avg(C)', 'index']
csv_writer.writerow(list)

run_list = range(0, 1000)
# run_list = [0,63,765,701,573,509,253,574,946,189,317,766,637,510,638,445,767,639,446,511,512,702,640,447,513,514,254,318,515,516,517,448,641,518,575,449,768,576,769,642,577,519,319,450,770,320,190,771,893,829,451,772,830,520,578,579,894,703,704,705,773,947,191,895,831,255,948,949,452,192,256,643,193,194,453,454,257,455,896,258,706,456,774,457,707,458,321,580,322,897,581,582,583,459,644,521,584,460,898,708,832,775,259,323,833,645,776,585,646,777,195,647,196,197,648,461,834,260,649,522,709,462,586,650,835,523,524,525,651,652,836,653,778,261,198,710,711,654,837,262,838,655,712,324,950,656,199,526,779,713,714,527,780,528,715,325,587,326,716,463,588,717,125,839,589,590,951,327,781,657,899,840,782,718,328,783,719,263,720,264,841,721,722,658,900,591,592,723,724,265,529,784,464,465,901,466,530,126,902,725,] 
# run_list = []
for i in run_list:
    file_json = open(os.path.join(name, 'json', op + '_' + str(i) + '.json'), 'r')
    line = file_json.readline()
    injson = json.loads(line)
    lat_tvm = injson['r'][0][0] # s
    if lat_tvm == 1e10: continue
    lat_tvm = round(lat_tvm * 1000, 4) # ms
    file_json.close()

    file_power = open(os.path.join(name, 'power_data', 'power_data_' + str(i) + '.txt'), 'r')
    energy = round(float(file_power.readline()), 3)
    power_avg = round(float(file_power.readline()), 2)
    power_peak = round(float(file_power.readline()), 2)
    file_power.close()

    file_temperature = open(os.path.join(name, 'temperature', 'temperature_' + str(i) + '.txt'), 'r')
    temp_list = file_temperature.readlines()
    temp_list = [int(temp.strip()) for temp in temp_list]
    temp_avg = round(sum(temp_list) / len(temp_list), 3)
    file_temperature.close()

    # file_nvprof_latency = open(os.path.join(name + '_ncu', op + '_' + str(i) + '_latency.txt'), 'r')
    # lat_nvprof = 10000
    # while True:
    #     text = file_nvprof_latency.readline()
    #     if text == "": break
    #     # print(text)
    #     text = text.split()
    #     if len(text) == 0: continue
    #     if text[-1].endswith("kernel0"):
    #         if text[-6][-2:]=='us': base = 1e-3
    #         elif text[-6][-2:]=='ms': base = 1
    #         else: print('error!')
    #         lat_nvprof = round(float(text[-6][:-2])*base, 5) # ms
    #         break
    # file_nvprof_latency.close()
    file_latency = open(os.path.join(name, 'lat', op + '_latency_' + str(i) + '.txt'), 'r')
    latency = round(float(file_latency.readline()), 4)
    file_latency.close()

    file_nvprof_metrics = open(os.path.join(name, 'ncu', op + '_metrics_' + str(i) + '.txt'), 'r')
    while True:
        text = file_nvprof_metrics.readline()
        if text == "": break
        # print(text)
        text = text.split()
        if len(text) == 0: continue
        if text[1] == "dram_read_transactions":
            dram_l = int(float(text[-1]))
        elif text[1] == "dram_write_transactions":
            dram_s = int(float(text[-1]))
        elif text[1] == "l2_read_transactions":
            l2_l = int(float(text[-1]))
        elif text[1] == "l2_write_transactions":
            l2_s = int(float(text[-1]))
        elif text[1] == "shared_load_transactions":
            shared_l = int(float(text[-1]))
        elif text[1] == "shared_store_transactions":
            shared_s = int(float(text[-1]))
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
    file_nvprof_metrics.close()
    # l2_l = l2_l + l2_base1 + l2_base2
    # l2_s = l2_s + l2_base1 + l2_base2
    # throughput_l2_l = throughput_l2_l + throughput_l2_base1 + throughput_l2_base2
    # throughput_l2_s = throughput_l2_s + throughput_l2_base1 + throughput_l2_base2


    list = [dram_l, dram_s, l2_l, l2_s, shared_l, shared_s, sm_efficiency, flop_efficiency, lat_tvm, latency, energy, power_peak, power_avg, temp_avg, i]
    # print(list)
    csv_writer.writerow(list)
f_csv.close()

# print("power_max: ", power_max, id_max)
# print("power_min: ", power_min, id_min)