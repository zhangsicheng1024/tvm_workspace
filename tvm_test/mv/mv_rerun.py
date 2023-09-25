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
import argparse

parser = argparse.ArgumentParser(description='mv kernel names')
parser.add_argument('--config1')
parser.add_argument('--config2')
args_top = parser.parse_args()

# Define the computation

@auto_scheduler.register_workload
def mv_layer(N, K):
    data = te.placeholder((1, K), name="data")
    kernel = te.placeholder((K, N), name="kernel")
    out = topi.nn.matmul(data, kernel, out_dtype="float32")
    return [data, kernel, out]

# Create the search task

target = tvm.target.Target("cuda")

N, K = 32768, 8192
# N, K = 49152, 12288
task = auto_scheduler.SearchTask(
    func=mv_layer, args=(N, K), target=target
)

log_file = args_top.config1
sch, args = task.apply_best(log_file)

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

# Generate .cu file
# print("Generate code:")
if args_top.config2:
    file_name = args_top.config2
    dev_module = func.imported_modules[0]
    with open(file_name, mode='w') as file:
        file.write(dev_module.get_source())
