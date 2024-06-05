#!/bin/bash -l
#SBATCH --job-name=3d
#SBATCH --qos=medium
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --output ./slurm_output/3d-%j.out

python main.py --batch_size $1 --type $2 --hidden_channels $3 --num_output_layers $4 --num_layers $5 --c_sol_emb_dim $6 --h_sol_emb_dim $7
