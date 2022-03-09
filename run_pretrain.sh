#!/bin/bash
#
#SBATCH --job-name=pretrain
#SBATCH --chdir /home/danmoral/PAC
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1


python addnl_scripts/pretrain/rot_pred_no_wandb.py --batch_size=16 --steps=101 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=resnet_seg --ckpt_freq=1 --pre_trained=True

