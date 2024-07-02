#!/bin/bash
#SBATCH -e /data/users1/cmccurdy5/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users1/cmccurdy5/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cmccurdy5@student.gsu.edu
#SBATCH --chdir=/data/users1/cmccurdy5/MISA-pytorch
#
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --array=0-2
#SBATCH --account=trends53c17
#SBATCH --job-name=MISAtorch
#SBATCH --verbose
#SBATCH --time=7200
#
#SBATCH --nodes=1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=32


sleep 5s
hostname
# --cpus-per-task=5
echo "before bashrc"
which conda
source ~/.bashrc
echo "before init"
which conda
. ~/init_miniconda3.sh
echo "before activate"
which conda
conda activate pt2

seed=(7 14 21)
w=('wpca' 'w0' 'w1')

SEED=${seed[$((SLURM_ARRAY_TASK_ID % 3))]}
echo $SEED
W=${w[$((SLURM_ARRAY_TASK_ID / 3))]}
echo $W
declare -i n_dataset=12
declare -i n_source=12
declare -i n_sample=32768
lrs=(0.01 0.001 0.0001)
configuration="/data/users1/cmccurdy5/MISA-pytorch/configs/sim-siva.yaml"
data_file="sim-siva_dataset"$n_dataset"_source"$n_source"_sample"$n_sample"_seed"$SEED".mat"
declare -i num_experiments=${#lrs[@]}
for ((i=0; i<num_experiments; i++)); do
    lr=${lrs[$i]}
    echo $lr
    python main.py -c "$configuration" -f "$data_file" -r results/MathPath2024/ -w "$W" -a -lr "$lr"
    sleep 5s
done