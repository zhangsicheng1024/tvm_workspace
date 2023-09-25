import os

name = 'mv_N32768K8192_ori_1000'
num = 948

json_dir = name + '_json'
ncu_dir = name + '_ncu'
if not os.path.exists(ncu_dir):
    os.mkdir(ncu_dir)

f=open("ncu.sh", 'w')
# query_list = 'dram__sectors_read.sum,dram__sectors_write.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed'
# query_list_mem_throughput = ',dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,lts__t_sectors_op_read.sum.per_second,lts__t_sectors_op_write.sum.per_second,lts__t_sectors_op_atom.sum.per_second,lts__t_sectors_op_red.sum.per_second,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second'
query_list = "dram_read_transactions,dram_write_transactions,l2_read_transactions,l2_write_transactions,shared_load_transactions,shared_store_transactions,flop_count_sp_fma,flop_sp_efficiency,sm_efficiency"
query_list_mem_throughput = ",dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,shared_load_throughput,shared_store_throughput"
for i in range(num):
    json_path = os.path.join(json_dir, 'mv_' + str(i) + '.json')
    metrics_path = os.path.join(ncu_dir, 'mv_' + str(i) + '_metrics')
    latency_path = os.path.join(ncu_dir, 'mv_' + str(i) + '_latency')
    # f_out.write('ncu --section SpeedOfLight --metrics \"' + query_list + query_list_mem_throughput + '\" --log-file mv_fds_300_ncu/mv_'+str(i)+" python mv_rerun.py --config1 ./mv_fds_300_json/mv_"+str(i)+".json\n")
    f.write('nvprof --metrics \"' + query_list + query_list_mem_throughput + '\" --log-file ' + metrics_path + ' python mv_rerun.py --config1 ' + json_path + '\n')
    f.write('nvprof --log-file ' + latency_path + ' python mv_rerun.py --config1 ' + json_path + '\n')

f.close()