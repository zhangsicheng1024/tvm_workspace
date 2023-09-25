import tvm
import numpy as np
from tvm.topi.testing import depthwise_conv2d_python_nchw

# print("start mv load run")

func = tvm.runtime.load_module("/data/tvm_test/tmp_buildtar/tmp_func.tar")
# print("load module done")

N, K = 4096, 1024
data_np = np.random.uniform(size=(1, K)).astype(np.float32)
weight_np = np.random.uniform(size=(K, N)).astype(np.float32)

dev = tvm.cuda()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty([1, N], device=dev)
func(data_tvm, weight_tvm, out_tvm)