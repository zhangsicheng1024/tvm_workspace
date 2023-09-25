import argparse
import random
import os

parser = argparse.ArgumentParser(description='kernel names')
parser.add_argument('--config')
args_top = parser.parse_args()

file_latency = open(args_top.config)
while True:
    text = file_latency.readline()
    if text == "": break
    # print(text)
    text = text.split()
    if len(text) == 0: continue
    if text[-1].endswith("kernel0"):
        if text[-6][-2:]=='us': base = 1e-6
        elif text[-6][-2:]=='ms': base = 1e-3
        else: print('error!')
        latency = round(float(text[-6][:-2])*base, 10) # s
        break

file_power=open("Power_data.txt")
lines = file_power.read()
if len(lines) == 0: # warmup timeout(very slow kernel), use power=400W as error
    print(400 * latency * 1000)
    print(400)
    print(400)
else:
    lines = lines.split()
    sum = 0.0
    power_peak = 0
    for i in lines:
        power = float(i)
        sum += power
        if power > power_peak: power_peak = power
    power_avg = sum / len(lines) # w

    energy = power_avg * latency * 1000 # mJ

    print(energy)
    print(power_avg)
    print(power_peak)

file_latency.close()
file_power.close()