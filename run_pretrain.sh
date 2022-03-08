#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --output=res_pretrain.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


# srun conda activate PAC2
# python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=5001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=expt2 --use_wandb
# python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=5001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=expt3 --use_wandb --ckpt_freq=1
# srun --gres=gpu:1 python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=601 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=test --use_wandb --ckpt_freq=1
srun --pty --gres=gpu:1 bash