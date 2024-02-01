#!/bin/bash

#SBATCH --job-name=re_test2_1     # Job name
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (process)
#SBATCH --mem=80G                       # Memory per node (e.g., 4 GB)
#SBATCH --time=144:00:00                  # Time limit (hh:mm:ss)


#SBATCH --export=all
#SBATCH --output=output/opt_test2_1.txt
#SBATCH --error=output/err_test2_1.txt

python -u toy2.py
