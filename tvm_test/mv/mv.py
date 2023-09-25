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
def mv_layer(N, K):
    # k = te.reduce_axis((0, K), "k")
    data = te.placeholder((1, K), name="data")
    kernel = te.placeholder((K, N), name="kernel")
    out = topi.nn.matmul(data, kernel, out_dtype="float32")
    # out = te.compute((1, N), lambda y: te.sum(data[0, k] * kernel[k, y], axis=k), name="out")
    return [data, kernel, out]
# N, K = 49152, 12288 # (4096, 1024) * 12
N, K = 32768, 8192 # (4096, 1024) * 8


# Create the search task
target = tvm.target.Target("cuda")

method = 'ori'
search_count = 1000
task = auto_scheduler.SearchTask(
    func=mv_layer, args=(N, K), target=target
)

log_file = 'mv_' + 'N' + str(N) + 'K' + str(K) + '_' + method + '_' + str(search_count) + '_2.json'
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=search_count,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run the search
task.tune(tune_option)

sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx

# Check correctness and evaluate performance
func = tvm.build(sch, args, target, name="mymv")

# Check correctness
data_np = np.random.uniform(size=(1, K)).astype(np.float32)
weight_np = np.random.uniform(size=(K, N)).astype(np.float32)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty([1, N], device=dev)
func(data_tvm, weight_tvm, out_tvm)
