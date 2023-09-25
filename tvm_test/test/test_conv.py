import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python

@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

if __name__ == "__main__":

    target = tvm.target.Target("cuda")

    # Use the last layer in ResNet-50
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
    task = auto_scheduler.SearchTask(
        func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
    )

    # Inspect the computational graph
    # print("Computational DAG:")
    # print(task.compute_dag)

    log_file = "conv2d.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=10,  # change this to 1000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    
    # task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    # Kill the measurement process
    del measure_ctx

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))