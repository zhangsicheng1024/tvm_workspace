import tvm
import numpy as np
from tvm.topi.testing import depthwise_conv2d_python_nchw
import argparse

# print("start dwconv load run")
parser = argparse.ArgumentParser(description='dwconv kernel names')
parser.add_argument('--config1')
parser.add_argument('--config2')
args_top = parser.parse_args()

# func = tvm.runtime.load_module("/data/tvm_test/tmp_buildtar/tmp_func.tar")
print('start load/build module')
func = tvm.runtime.load_module(args_top.config1) # <class 'tvm.runtime.module.Module'>
# print(func.get_source())

N, H, W, CM, CI, KH, KW, strides, padding = 128, 28, 28, 1, 128, 3, 3, (1, 1), (1, 1)
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CI, CM, KH, KW)).astype(np.float32)
# conv_np = depthwise_conv2d_python_nchw(data_np, weight_np, strides, padding)
conv_np = np.random.uniform(size=(N, CM * CI, H, W)).astype(np.float32)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
print('call func')
func(data_tvm, weight_tvm, out_tvm)

print('type func', type(func))
print(func)
dev_module = func.imported_modules[0]
print('type dev_module', type(dev_module))
print(dev_module)
print(dev_module.get_source())

if args_top.config2:
    file_name = args_top.config2
    dev_module = func.imported_modules[0]
    with open(file_name, mode='w') as file:
        file.write(dev_module.get_source())