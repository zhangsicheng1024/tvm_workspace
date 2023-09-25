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
import argparse

parser = argparse.ArgumentParser(description='dwconv kernel names')
parser.add_argument('--config1')
parser.add_argument('--config2')
args_top = parser.parse_args()

# Define the computation
@auto_scheduler.register_workload
def dwconv2d_layer(N, H, W, CM, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CI, CM, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

# Create the search task

target = tvm.target.Target("cuda")

N, H, W, CM, CI, KH, KW, strides, padding = 128, 28, 28, 1, 128, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.SearchTask(
    func=dwconv2d_layer, args=(N, H, W, CM, CI, KH, KW, strides, padding), target=target
)

log_file = args_top.config1

# Apply the best schedule
sch, args = task.apply_best(log_file)

# Check correctness and evaluate performance

# print('start load/build module')
# sch: <class 'tvm.te.schedule.Schedule'>
func = tvm.build(sch, args, target, name="mydwconv") # <class 'tvm.driver.build_module.OperatorModule'>

# Check correctness

data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CI, CM, KH, KW)).astype(np.float32)
conv_np = np.random.uniform(size=(N, CM * CI, H, W)).astype(np.float32)
# conv_np = depthwise_conv2d_python_nchw(data_np, weight_np, strides, padding)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

func(data_tvm, weight_tvm, out_tvm)

if args_top.config2:
    file_name = args_top.config2
    dev_module = func.imported_modules[0]
    with open(file_name, mode='w') as file:
        file.write(dev_module.get_source())
