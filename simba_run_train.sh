#!/bin/bash
#
#SBATCH --job-name=train_PAC
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30000
#SBATCH --time=06:00:00


python main.py --steps=50001 --dataset=multi --source=real --target=sketch --backbone=expts/rot_pred/checkpoint.pth.tar --vat_tw=0 --expt_name=simba_1

