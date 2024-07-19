#!/bin/bash
#SBATCH -e /data/users1/cmccurdy5/MISA-pytorch/slurm_logs/error%A-%a.err
#SBATCH -o /data/users1/cmccurdy5/MISA-pytorch/slurm_logs/out%A-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cmccurdy5@student.gsu.edu
#SBATCH --chdir=/data/users1/cmccurdy5/MISA-pytorch
#
#SBATCH -p qTRDGPU
#SBATCH --gres=gpu:A40:1
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
source ~/.bashrc
. ~/init_miniconda3.sh
conda activate pt2

# seed=(7 14 21)
# w=('wpca' 'w0' 'w1')

declare -i SEED=7
echo $SEED
W="wpca"
echo $W

declare -i n_dataset=100
declare -i n_source=12
declare -i n_sample=32768
lrs=(0.01)
batch_size=(316)
patience=(10)
gpu=("A100")
#Adam optimizer parameters
adam_params=(0) #0 sets foreach and fused to false, 1 sets foreach=true and fused=false, 2 for foreach=false and fused=true
beta1=(0.7)
beta2=(0.65)

experimenter="$USER"
configuration="/data/users1/cmccurdy5/MISA-pytorch/configs/sim-siva.yaml"
data_file="sim-siva_dataset"$n_dataset"_source"$n_source"_sample"$n_sample"_seed"$SEED".mat"
declare -i num_experiments=${#lrs[@]}
for ((i=0; i<num_experiments; i++)); do
    python main.py -c "$configuration" -f "$data_file" -r results/MathPath2024/ -w "$W" -a -lr "${lrs[$i]}" -b1 "${beta1[$i]}" -b2 "${beta2[$i]}" -bs "${batch_size[$i]}" -e "$experimenter" -ff "${adam_params[$i]}" -p "${patience[$i]}" -s "$SEED" -g "${gpu[$i]}" &
    sleep 5s
    wait 
done