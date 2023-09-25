# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.topi.testing import depthwise_conv2d_python_nchw
from tvm.contrib import utils

# Define the computation
@auto_scheduler.register_workload
def dwconv2d_layer(N, H, W, CM, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CI, CM, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

# Create the search task

target = tvm.target.Target("cuda")

# Mobile net: 56 * 56 * 128
# Here: 56 * 56 * 512, N = 128
method = 'energy'
search_count = 1000
N, H, W, CM, CI, KH, KW, strides, padding = 128, 28, 28, 1, 128, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.SearchTask(
    func=dwconv2d_layer, args=(N, H, W, CM, CI, KH, KW, strides, padding), target=target
)

log_file = 'dwconv_' + 'N' + str(N) + 'C' + str(CI) + 'H' + str(H) + 'W' + str(W) + '_' + method + '_' + str(search_count) + '_2.json'
print('log file =', log_file)
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=search_count,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run the search

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx

# Check correctness and evaluate performance

func = tvm.build(sch, args, target, name="mydwconv")

# Check correctness

data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CI, CM, KH, KW)).astype(np.float32)
conv_np = depthwise_conv2d_python_nchw(data_np, weight_np, strides, padding)
print('conv_np shape:', conv_np.shape)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
func(data_tvm, weight_tvm, out_tvm)
