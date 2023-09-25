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

parser = argparse.ArgumentParser(description='mv kernel names')
parser.add_argument('--config1')
parser.add_argument('--config2')
args_top = parser.parse_args()


# Define the computation
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    # out = topi.nn.relu(conv + bias)
    # return [data, kernel, bias, out]
    return [data, kernel, conv]

target = tvm.target.Target("cuda")

N, H, W, CO, CI, KH, KW, strides, padding = 128, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1)
# N, H, W, CO, CI, KH, KW, strides, padding = 64, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1) # ResNet-50
# N, H, W, CO, CI, KH, KW, strides, padding = 128, 19, 19, 2560, 640, 3, 3, (1, 1), (2, 2) #effientnet-B7
task = auto_scheduler.SearchTask(
    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
)

log_file = args_top.config1

sch, args = task.apply_best(log_file)

func = tvm.build(sch, args, target, name="myconv")

# Check correctness

data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
# conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
conv_np = np.random.uniform(size=(N, CO, H, W)).astype(np.float32)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
conv_tvm = tvm.nd.empty(conv_np.shape, device=dev)

func(data_tvm, weight_tvm, conv_tvm)

# Generate .cu file

if args_top.config2:
    file_name = args_top.config2
    dev_module = func.imported_modules[0]
    with open(file_name, mode='w') as file:
        file.write(dev_module.get_source())
