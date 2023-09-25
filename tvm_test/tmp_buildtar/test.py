import re
import os

gpu_id = os.environ['CUDA_VISIBLE_DEVICES']
print(gpu_id == '3')
