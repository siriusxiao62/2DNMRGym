#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_out/eval_%j.out
#SBATCH --gres=gpu:1

python eval_model.py 
