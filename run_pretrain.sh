#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --output=res_pretrain.txt
#
#SBATCH --ntasks=3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


srun hostname
srun python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=1001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred 