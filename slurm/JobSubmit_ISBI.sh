#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --mem=44g
#SBATCH --array=0-9
#SBATCH -t 7200
#SBATCH -J MISA-torch
#SBATCH -e ./err/err%A-%a.err
#SBATCH -o ./out/out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xli993@gatech.edu

sleep 5s

source /home/users/xli77/anaconda3/bin/activate
# source /home/users/ga20055/anaconda3/bin/activate
conda activate pixl

seed=(7 14 21)
w=('wpca' 'w0' 'w1')

SEED=${seed[$((SLURM_ARRAY_TASK_ID % 3))]}
echo $SEED
W=${w[$((SLURM_ARRAY_TASK_ID / 3))]}
echo $W

cd /data/users2/xli/MISA-pytorch
python run_main.py -s $SEED -w $W

sleep 5s