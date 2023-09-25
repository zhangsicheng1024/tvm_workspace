import random
import time
import json
import subprocess
import psutil
import os
from shutil import copyfile

name = 'matmul_M1024N4096K1024_energy_pc150+300_1000'
gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
assert gpu_id != ""
workdir = './gpu' + gpu_id + '/'
op = name.split('_')[0]

if not os.path.exists(name): os.mkdir(name)
json_dir = name + '/json'
if not os.path.exists(json_dir): os.mkdir(json_dir)
tiling_dir = name + '/tiling'
if not os.path.exists(tiling_dir): os.mkdir(tiling_dir)
cu_dir = name + '/cu'
if not os.path.exists(cu_dir): os.mkdir(cu_dir)
gridblk_dir = name + '/gridblk'
if not os.path.exists(gridblk_dir): os.mkdir(gridblk_dir)
lat_dir = name + '/lat'
if not os.path.exists(lat_dir): os.mkdir(lat_dir)
ncu_dir = name + '/ncu'
if not os.path.exists(ncu_dir): os.mkdir(ncu_dir)
kernel_cu_dir = name + '/kernel_cu'
if not os.path.exists(kernel_cu_dir): os.mkdir(kernel_cu_dir)
power_data_dir = name + '/power_data'
if not os.path.exists(power_data_dir): os.mkdir(power_data_dir)
power_sample_dir = name + '/power_sample'
if not os.path.exists(power_sample_dir): os.mkdir(power_sample_dir)
temperature_dir = name + '/temperature'
if not os.path.exists(temperature_dir): os.mkdir(temperature_dir)

# json preprocess
json_preprocess = True
if json_preprocess:
    json_in = open(name + '.json', 'r')
    lines = json_in.read().split('\n')

    for i, line in enumerate(lines):
        json_out = open(os.path.join(json_dir, op + '_' + str(i) + '.json'), 'w')
        json_out.write(lines[i])
        json_out.close()
    json_in.close()

# run_list = range(0, 1000)
# M1024N4096K1024
# run_list = [315,187,188,251,189,124,190,125,252,379,380,126,253,316,254,255,256,443,257,317,444,191,192,193,0,258,259,127,318,194,260,195,261,262,263,196,128,129,130,131,319,197,264,320,265,198,199,266,890,891,321,698,200,445,446,507,447,699,634,571,381,635,382,267,201,762,202,203,383,204,384,268,205,508,509,206,636,385,207,510,386,387,208,132,511,209,210,637,133,388,572,389,211,512,269,448,513,134,638,514,515,826,212,449,573,322,827,390,213,391,323,270,945,324,574,450,575,271,451,700,576,392,639,393,640,701,577,394,214,702,395,516,517,578,518,519,703,215,579,580,763,135,892,893,581,641,642,704,452,643,582,453,454,583,764,136,455,584,828,216,520,456,521,396,137,522,138,894,585,457,829,523,524,272,325,217,458,139,895,459,525,644,830,397,460,273,765,586,946,274,766,587,767,831,275,218,645,947,526,219,]
# M8192N4096K1024
# run_list = [441,249,442,250,377,251,443,378,697,252,633,444,944,185,634,635,445,379,446,447,253,636,313,448,569,449,824,450,380,254,314,315,381,316,505,382,255,637,256,825,317,945,946,888,638,383,761,384,451,452,318,639,319,320,321,322,257,947,385,453,323,948,949,324,325,386,387,258,388,698,762,763,389,390,391,392,454,699,889,506,764,700,640,701,393,326,259,890,394,507,395,570,950,327,571,702,891,703,328,892,260,893,894,641,455,572,456,396,397,398,765,766,261,457,262,399,458,263,329,573,459,400,264,330,574,575,642,460,331,508,332,509,461,462,401,463,402,951,403,333,643,464,465,644,576,265,645,646,404,186,405,826,827,334,266,577,647,648,466,828,335,649,467,895,578,336,337,650,406,767,704,338,267,510,268,187,952,768,468,705,469,511,269,188,706,579,512,707,339,407,340,270,470,513,471,189,341,580,190,271,]
# M4096N4096K1024
# run_list = [822,694,886,943,695,758,944,759,823,696,887,760,697,888,889,698,761,890,891,945,892,824,762,374,825,763,630,764,438,946,826,631,893,699,827,700,375,828,894,701,502,895,896,897,702,703,503,765,947,632,898,899,900,633,704,901,829,830,902,948,903,831,376,766,310,832,767,439,904,768,246,905,769,949,770,833,247,705,771,906,834,706,504,835,634,836,635,505,772,636,566,707,708,837,838,709,839,907,840,377,567,908,637,710,841,248,711,249,506,773,909,568,507,774,311,440,910,378,712,638,842,250,911,775,379,508,441,713,950,509,714,912,251,776,777,639,640,380,312,442,778,252,641,443,715,510,444,642,716,951,569,381,511,570,313,913,779,643,952,182,314,315,780,717,781,843,844,914,445,316,915,571,512,572,382,317,383,916,845,384,573,513,644,782,645,846,514,646,917,647,318,918,515,446,516,574,783,385,253,784,]
# M1024N4096K1024_2
# run_list = [570,442,571,890,314,250,572,634,122,378,443,315,945,506,251,635,444,636,186,187,252,316,253,254,573,507,826,891,188,762,827,828,829,255,946,892,893,189,894,895,637,698,123,699,317,256,896,257,445,124,379,446,447,897,190,318,830,319,258,508,125,700,763,509,574,575,510,898,764,380,259,899,448,765,511,320,321,322,947,512,701,576,831,702,638,513,832,191,703,704,381,766,705,706,833,707,948,639,323,900,834,901,767,708,709,514,640,949,382,260,577,515,449,768,641,578,642,579,902,835,836,580,950,710,581,450,516,324,837,838,261,582,192,903,583,839,643,711,325,517,904,518,644,769,712,840,770,771,519,520,584,451,383,452,645,326,646,713,951,841,647,905,521,327,193,842,714,648,585,906,453,649,715,454,952,716,772,773,843,907,586,455,384,774,587,588,650,651,522,775,652,523,844,385,776,777,653,778,717,328,]                                                                                
# M1024N4096K1024_pc150+150
run_list = [61,313,62,761,762,763,697,698,569,570,571,764,505,633,699,765,572,766,573,634,441,574,767,700,768,635,636,701,769,442,637,638,377,770,506,507,639,640,771,772,443,641,314,773,774,642,575,249,576,702,643,703,644,508,645,775,509,577,776,444,777,778,445,121,646,647,779,315,510,780,578,781,704,705,579,706,648,649,580,782,707,783,511,581,708,582,512,583,650,784,446,785,513,709,786,447,514,584,585,651,787,316,710,711,788,652,712,185,713,789,586,587,515,790,714,653,448,654,655,715,656,317,378,657,250,658,791,792,516,716,449,517,717,450,659,518,519,318,451,452,319,588,251,660,520,186,718,252,453,589,793,320,454,661,590,662,321,794,663,795,796,379,664,719,455,797,456,665,798,720,799,187,521,522,122,253,380,523,666,524,457,525,458,526,591,254,322,592,800,527,667,381,593,801,802,528,188,529,594,530,]
# M1024N4096K1024_zyj
# run_list = [3,12,26,29,33,54,56,57,59,65,70,72,75,82,85,90,96,107,117,119,122,126,189,190,254,446]
run_list = [55,56,57,58,59,60]

def get_latency(repeat_warmpup, repeat_run):
    # get gridblk and declaration
    file_gridblk = open(path_gridblk, 'r')
    grid = file_gridblk.readline().split()
    blk = file_gridblk.readline().split()
    file_gridblk.close()

    file_cu = open(path_cu, 'r')
    while True:
        line = file_cu.readline()
        if not line: break
        line = line.split()
        if len(line) == 0: continue
        if(line[0] == 'extern'):
            declare = ' '.join(line[:-1]) + ';\n'
            break
    file_cu.close()
    
    # generate kernel cuda for latency
    command_copy_kernel = 'cp ' + path_kernel_cu_lat_template + ' ' + path_kernel_cu_lat
    print(command_copy_kernel)
    os.system(command_copy_kernel)

    file_kernel_cu_lat_template = open(path_kernel_cu_lat_template, 'r')
    content = file_kernel_cu_lat_template.read()
    file_kernel_cu_lat_template.close()

    file_kernel_cu_lat = open(path_kernel_cu_lat, 'w')
    file_kernel_cu_lat.write('#define REPEAT_WARMUP ' + str(repeat_warmpup) + '\n')
    file_kernel_cu_lat.write('#define REPEAT_RUN ' + str(repeat_run) + '\n')
    file_kernel_cu_lat.write('dim3 dimGrid(' + grid[0] + ', ' + grid[1] + ', ' + grid[2] + ');\n')
    file_kernel_cu_lat.write('dim3 dimBlock(' + blk[0] + ', ' + blk[1] + ', ' + blk[2] + ');\n')
    file_kernel_cu_lat.write(declare + '\n')
    file_kernel_cu_lat.write(content)
    file_kernel_cu_lat.close()

    # generate latency, warmup + latency(run)
    command_make = 'nvcc -o ' + path_kernel_o + ' ' + path_cu + ' ' +  path_kernel_cu_lat + ' ' + HEADERNVMLAPI + ' ' + INCLUDEPROG
    print(command_make)
    os.system(command_make)

    command_run = path_kernel_o + ' > ' + path_latency
    print(command_run)
    os.system(command_run)
    file_latency = open(path_latency, 'r')
    latency = float(file_latency.readline().strip()) / 1000.0 # s
    file_latency.close()
    return latency



for index in run_list: 
    print('index =', index)
    path_json = os.path.join(json_dir, op + '_' + str(index) + '.json')
    path_rerun = os.path.join(workdir, op + '_rerun.py')
    path_tiling = os.path.join(workdir, op + '_tiling.txt')
    path_cu = os.path.join(workdir, op + '.cu')
    path_gridblk = os.path.join(workdir, op + '_gridblk.txt')
    path_latency = os.path.join(workdir, op + '_latency.txt')
    path_metrics = os.path.join(workdir, op + '_metrics.txt')
    path_kernel_cu_lat_template = os.path.join(workdir, op + '_kernel_lat_template.cu')
    path_kernel_cu_lat = os.path.join(workdir, op + '_kernel_lat.cu')
    path_kernel_cu_ene_template = os.path.join(workdir, op + '_kernel_ene_template.cu')
    path_kernel_cu_ene = os.path.join(workdir, op + '_kernel_ene.cu')
    path_kernel_o = os.path.join(workdir, op + '_o')

    HEADERNVMLAPI = '-lnvidia-ml -L/usr/local/cuda-11.0/lib64 -lcuda -I/usr/local/cuda-11.0/include -lpthread -I/workspace/tvm_test/nvml-power'
    INCLUDEPROG = '/workspace/tvm_test/nvml-power/nvmlPower' + gpu_id + '.cpp'

    # invalid, continue
    with open(path_json) as json_in:
        line = json_in.readline()
    injson = json.loads(line)
    if injson['r'][0][0] >= 1e+10 or injson['r'][1] != 0:
        print('Invalid result, continue')
        continue

    # # generate tiling
    # command_tiling = 'python ' + 'get_tiling.py' + ' --json ' + path_json + ' > ' + path_tiling
    # print(command_tiling)
    # os.system(command_tiling)
    # copyfile(path_tiling, os.path.join(tiling_dir, op + '_' + str(index) + '_tiling.txt'))
    # continue

    # generate gridblk and cuda src
    command_gridblk_cu = 'python ' + path_rerun + ' --config1 ' + path_json + ' --config2 ' + path_cu + ' > ' + path_gridblk
    print(command_gridblk_cu)
    os.system(command_gridblk_cu)
    copyfile(path_cu, os.path.join(cu_dir, op + '_' + str(index) + '.cu'))
    copyfile(path_gridblk, os.path.join(gridblk_dir, op + '_gridblk_' + str(index) + '.txt'))

    # generate nvprof metrics
    # dram_read(write)_transactions(bytes), shared_load(store)_transactions, local_load(store)_transactions
    query_list = "dram_read_transactions,dram_write_transactions,l2_read_transactions,l2_write_transactions,shared_load_transactions,shared_store_transactions,flop_count_sp_fma,flop_sp_efficiency,sm_efficiency"
    query_list_mem_throughput = ",dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,shared_load_throughput,shared_store_throughput"
    # command_metrics = 'nvprof --metrics \"' + query_list + '\" --log-file ' + path_metrics + ' python ' + path_rerun + ' --config1 ' + path_json + ' > /dev/null'
    command_metrics = 'nvprof --metrics all --log-file ' + path_metrics + ' python ' + path_rerun + ' --config1 ' + path_json + ' > /dev/null'
    print(command_metrics)
    os.system(command_metrics)
    copyfile(path_metrics, os.path.join(ncu_dir, op + '_metrics_' + str(index) + '.txt'))

    '''
    # generate latency, use nvprof
    command_latency = 'nvprof --log-file ' + path_latency + ' python ' + path_rerun + ' --config1 ' + path_json + ' > /dev/null'
    print(command_latency)
    os.system(command_latency)
    copyfile(path_latency, os.path.join(ncu_dir, op + '_' + str(index) + '_latency.txt'))

    # get latency
    get_latency = False
    file_latency = open(path_latency)
    while True:
        text = file_latency.readline()
        if text == "": break
        # print(text)
        text = text.split()
        if len(text) == 0: continue
        if text[-1].endswith("kernel0"):
            if text[-6][-2:]=='us': base = 1e-6
            elif text[-6][-2:]=='ms': base = 1e-3
            else: print('error!')
            latency = float(text[-6][:-2])*base # s
            get_latency = True
            break
    file_latency.close()
    if get_latency: 
        print('latency =', latency)
    else:
        print('fail to get latency')
        latency = 10.0 + random.random() # 10s, as default timeout
    '''

    # generate latency
    latency_rough = get_latency(0, 1)       # warmup=0, run=1
    print('latency_rough =', latency_rough)
    warmup_time = 2 # s
    warmup_repeat = int(warmup_time / latency_rough)
    latency = get_latency(warmup_repeat, 1) # warmup=2s, run=1
    copyfile(path_latency, os.path.join(lat_dir, op + '_latency_' + str(index) + '.txt'))
    print('latency =', latency)

    # get gridblk and declaration
    file_gridblk = open(path_gridblk, 'r')
    grid = file_gridblk.readline().split()
    blk = file_gridblk.readline().split()
    file_gridblk.close()

    file_cu = open(path_cu, 'r')
    while True:
        line = file_cu.readline()
        if not line: break
        line = line.split()
        if len(line) == 0: continue
        if(line[0] == 'extern'):
            declare = ' '.join(line[:-1]) + ';\n'
            break
    file_cu.close()

    # generate kernel cuda for energy
    command_copy_kernel = 'cp ' + path_kernel_cu_ene_template + ' ' + path_kernel_cu_ene
    print(command_copy_kernel)
    os.system(command_copy_kernel)

    RUNTIME = 20 # s
    repeat_times = int(RUNTIME / latency)
    print('add gridblk & repeat, repeat = ', repeat_times)

    file_kernel_cu_ene_template = open(path_kernel_cu_ene_template, 'r')
    content = file_kernel_cu_ene_template.read()
    file_kernel_cu_ene_template.close()

    file_kernel_cu_ene = open(path_kernel_cu_ene, 'w')
    file_kernel_cu_ene.write('#define REPEAT ' + str(repeat_times) + '\n')
    file_kernel_cu_ene.write('dim3 dimGrid(' + grid[0] + ', ' + grid[1] + ', ' + grid[2] + ');\n')
    file_kernel_cu_ene.write('dim3 dimBlock(' + blk[0] + ', ' + blk[1] + ', ' + blk[2] + ');\n')
    file_kernel_cu_ene.write(declare + '\n')
    file_kernel_cu_ene.write(content)
    file_kernel_cu_ene.close()

    copyfile(path_kernel_cu_ene, os.path.join(kernel_cu_dir, op + '_kernel_' + str(index) + '.cu'))
    
    # build
    command_make = 'nvcc -o ' + path_kernel_o + ' ' + path_cu + ' ' +  path_kernel_cu_ene + ' ' + HEADERNVMLAPI + ' ' + INCLUDEPROG
    print(command_make)
    os.system(command_make)

    # run
    print(path_kernel_o)
    # TIMEOUT = RUNTIME * 2 # s
    # subp = subprocess.Popen([path_kernel_o])
    # p = psutil.Process(subp.pid)
    # try:
    #     p.wait(timeout=TIMEOUT)
    #     print('run finish')
    # except psutil.TimeoutExpired:
    #     p.kill()
    #     print('run timeout(' + str(TIMEOUT) + 's), kill')
    os.system(path_kernel_o)

    # collect power result & energy
    file_power_sample=open('Power_data' + gpu_id + '.txt')
    lines = file_power_sample.read()
    if len(lines) == 0: # warmup timeout(very slow kernel), use power=400W as error
        energy = 400 * latency * 1000
    else:
        lines = lines.split()
        sum = 0.0
        power_peak = 0
        for i in lines:
            power = float(i)
            sum += power
            if power > power_peak: power_peak = power
        power_avg = sum / len(lines) # w
        energy = power_avg * latency * 1000 # mJ
    file_power_sample.close()
    copyfile('Power_data' + gpu_id + '.txt', os.path.join(power_sample_dir, 'power_sample_' + str(index) + '.txt'))
    copyfile('Temperature_data' + gpu_id + '.txt', os.path.join(temperature_dir, 'temperature_' + str(index) + '.txt'))
    file_power_data = open(os.path.join(power_data_dir, 'power_data_' + str(index) + '.txt'), 'w')
    file_power_data.write(str(energy) + '\n')
    file_power_data.write(str(power_avg) + '\n')
    file_power_data.write(str(power_peak) + '\n')
    file_power_data.close()
    print('latency = %f, energy = %f, power_peak = %f, power_avg = %f' % (latency, energy, power_peak, power_avg))
    print()
