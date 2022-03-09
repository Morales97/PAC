#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=5001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=simba_slurm --ckpt_freq=1 --pre_trained=True

