#!/bin/bash -l
#SBATCH --job-name=eval
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:V100:3
#SBATCH --output ./eval-%j.out

conda activate /work/yunruili/anaconda_env/diffusion
python eval_model.py 
