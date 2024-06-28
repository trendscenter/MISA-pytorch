import os
import argparse

seed_list = [7, 14, 21]
sample_list = [64, 128, 512, 256, 1024, 4096, 16384, 32768]
dataset_list = [2, 3, 4, 12, 32, 100]
source_list = [8, 12, 32, 100]

old_sample_list = [64, 256, 1024, 4096, 16384, 32768]
old_dataset_list = [2, 12, 32, 100]
old_source_list = [12, 32, 100]

parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--seed', type=int, default=7, help='Random seed')
parser.add_argument('-w', '--weights', type=str, default='w0', help='Name of weighted matrix W in the dataset')
parser.add_argument('-r', '--run', type=str, default='/run', help='Result dir')

args = parser.parse_args()
seed = args.seed
w = args.weights

for n_dataset in dataset_list:
    for n_source in source_list:
        if (n_dataset==32 and n_source==100) or (n_dataset==100 and n_source==32) or (n_dataset==100 and n_source==100):
            continue
        for n_sample in sample_list:
            if n_source > n_sample:
                continue
            if (n_dataset in old_dataset_list) and (n_source in old_source_list) and (n_sample in old_sample_list):
                continue
            cmd = f"python3 main.py -c /data/users2/xli/MISA-pytorch/configs/sim-iva.yaml -f sim-siva_dataset{n_dataset}_source{n_source}_sample{n_sample}_seed{seed}.mat -r run_iva_0423 -w {w} -a"
            os.system(cmd)