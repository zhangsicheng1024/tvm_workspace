# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib import utils

# Define the computation
@auto_scheduler.register_workload
def matmul_layer(M, N, K):
    data = te.placeholder((M, K), name="data")
    kernel = te.placeholder((K, N), name="kernel")
    out = topi.nn.matmul(data, kernel, out_dtype="float32")
    return [data, kernel, out]

# Create the search task
target = tvm.target.Target("cuda")

method = 'test'
search_count = 100
# M, N, K = 65536, 4096, 1024
M, N, K = 65536, 4096, 1024
task = auto_scheduler.SearchTask(
    func=matmul_layer, args=(M, N, K), target=target
)

log_file = 'matmul_' + 'M' + str(M) + 'N' + str(N) + 'K' + str(K) + '_' + method + '_' + str(search_count) + '.json'
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=search_count,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)

# Apply the best schedule
sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx


func = tvm.build(sch, args, target, name="mymatmul")

# Check correctness

data_np = np.random.uniform(size=(M, K)).astype(np.float32)
weight_np = np.random.uniform(size=(K, N)).astype(np.float32)
dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty([M, N], device=dev)
func(data_tvm, weight_tvm, out_tvm)
