#!/bin/bash -l
#SBATCH --job-name=2d
#SBATCH --account=pengyu-lab
#SBATCH --partition=pengyu-gpu
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --output ./slurm_output/2d-%j.out

conda activate /work/yunruili/anaconda_env/diffusion
python main.py --batch_size $1 --type $2 --hidden_channels $3 --num_layers $4 --c_sol_emb_dim $5 --h_sol_emb_dim $6
