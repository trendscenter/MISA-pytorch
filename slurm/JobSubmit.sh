#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH,qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --mem=44g
#SBATCH --array=0
#SBATCH -t 7200
#SBATCH -J MISA-pytorch
#SBATCH -e /data/users2/dkhosravinezhad1/MISA-batch/MISA-error
#SBATCH -o /data/users2/dkhosravinezhad1/MISA-batch/MISA-output
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dkhosravinezhad1@student.gsu.edu

sleep 5s

source /data/users2/dkhosravinezhad1/anaconda3/bin/activate
conda activate ipy

cd /data/users2/dkhosravinezhad1/MISA-pytorch
python main.py --run run/$((SLURM_ARRAY_TASK_ID))/

sleep 5s