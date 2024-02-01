#!/bin/bash

#SBATCH --job-name=toy3D     # Job name
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=compute
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (process)
#SBATCH --mem=90G                          # Memory per node (e.g., 4 GB)
#SBATCH --time=144:00:00                  # Time limit (hh:mm:ss)


#SBATCH --export=all
#SBATCH --output=output/opt_toy3D.txt
#SBATCH --error=output/err_toy3D.txt

python -u toy.py
