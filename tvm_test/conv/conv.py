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


######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a convolution layer.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    # out = topi.nn.relu(conv + bias)
    # return [data, kernel, bias, out]
    return [data, kernel, conv]

# Create the search task

target = tvm.target.Target("cuda")

method = 'ori'
search_count = 500
N, H, W, CO, CI, KH, KW, strides, padding = 128, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1)
# N, H, W, CO, CI, KH, KW, strides, padding = 64, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1) # ResNet-50
# N, H, W, CO, CI, KH, KW, strides, padding = 128, 19, 19, 2560, 640, 3, 3, (1, 1), (2, 2) #effientnet-B7
task = auto_scheduler.SearchTask(
    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
)

log_file = 'conv_' + 'N' + str(N) + 'C' + str(CO) + 'H' + str(H) + 'W' + str(W) + '_' + method + '_' + str(search_count) + '.json'
print('log_file =', log_file)
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

del measure_ctx

func = tvm.build(sch, args, target, name="myconv")

# Check correctness
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
conv_tvm = tvm.nd.empty(conv_np.shape, device=dev)
func(data_tvm, weight_tvm, conv_tvm)
