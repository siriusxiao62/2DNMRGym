#!/bin/bash -l
#SBATCH --job-name=eval
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:V100:3
#SBATCH --output ./eval-%j.out

python eval_model.py 
