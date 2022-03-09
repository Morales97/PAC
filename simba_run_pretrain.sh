#!/bin/bash
#
#SBATCH --job-name=pretrain
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4096
#SBATCH --time=06:00:00

python addnl_scripts/pretrain/rot_pred.py --batch_size=16 --steps=5001 --dataset=multi --source=real --target=sketch --save_dir=expts/rot_pred --expt_name=simba_slurm_2 --ckpt_freq=1 --pre_trained=True

