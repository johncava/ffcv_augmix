#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -n 1  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 1-00:00:00   # time in d-hh:mm:ss
#SBATCH -p serial       # partition 
#SBATCH -q normal       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=jcava@asu.edu # Mail-to address
#SBATCH --export=NONE   # Purge the job-submitting shell environment

module anaconda/py3
source activate FFCV

export IMAGENET_DIR=/scratch/jcava/imagenet-dataset-v2/
export WRITE_DIR=/scratch/jcava/imagenet_ffcv/
./write_imagenet.sh 400 1.00 50
