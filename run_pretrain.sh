#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --output=res_pretrain.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0


srun hostname
srun conda activate PAC2
srun python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=601 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --num_workers=1 
# python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=601 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=expt1 --use_wandb=True