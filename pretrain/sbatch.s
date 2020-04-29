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

module purge
module load cuda/10.0.130
module load cudnn/10.0v7.4.2.24

conda activate pytorch


cd /scratch/jd4138/dl_proj/pretrain

python pirl_train.py --num-scene 106 --model-type res50 --batch-size 128 --lr 0.005 --count-negatives 40000
