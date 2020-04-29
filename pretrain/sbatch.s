#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=pretrain
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jd4138@nyu.edu
#SBATCH --output=slurm_%j.out





. ~/.bashrc
module load anaconda3/5.3.1
module load cudnn/10.1v7.6.5.32
module load cuda/10.1.105

conda activate pytorch


cd /scratch/jd4138/dl_proj/pretrain
import torch
python3 pirl_train.py --num-scene 106 --model-type res50 --batch-size 128 --lr 0.005 --count-negatives 40000
