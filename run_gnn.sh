#!/bin/bash
#SBATCH --job-name=notransformer
#SBATCH --output=slurm_out/2dgnn_%j.out
#SBATCH --gres=gpu:1

python main.py --notransformer  --n_epoch 150 --type $1 --hidden_channels $2 --num_layers $3 --c_sol_emb_dim $4 --h_sol_emb_dim $5
