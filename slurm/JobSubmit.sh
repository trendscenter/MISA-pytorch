#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH,qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --mem=44g
#SBATCH -t 7200
#SBATCH -J pixl
#SBATCH -e err%A-%a.err
#SBATCH -o out%A-%a.out
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xli993@gatech.edu

sleep 5s

source /home/users/ga20055/anaconda3/bin/activate
conda activate pixl

cd /data/users1/xinhui/fusion
python main.py 

sleep 5s