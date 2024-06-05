#!/bin/bash -l
#SBATCH --job-name=3d
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:RTX2:1
#SBATCH --output ./slurm_output/schnet-%j.out

python main.py --batch_size $1 --type $2 --hidden_channels $3 --num_filters $4 --num_gaussians $5 --num_layers $6 --c_sol_emb_dim $7 --h_sol_emb_dim $8
