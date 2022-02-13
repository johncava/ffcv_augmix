#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -n 12  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-01:00:00   # time in d-hh:mm:ss
#SBATCH -p gpu       # partition
#SBATCH --gres=gpu:1
#SBATCH -C V100
#SBATCH -q wildfire       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # Mail-to address
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module load anaconda/py3
source activate ffcv.cuda.11.3

export CUDA_VISIBLE_DEVICES=0

python train_imagenet_augmix.py --config-file rn18_configs/rn18_16_epochs.yaml \
    --data.train_dataset=/scratch/jcava/imagenet_ffcv/train_400_1.00_50.ffcv \
    --data.val_dataset=/scratch/jcava/imagenet_ffcv/val_400_1.00_50.ffcv \
    --data.num_workers=12 --data.in_memory=0 \
    --logging.folder=./
