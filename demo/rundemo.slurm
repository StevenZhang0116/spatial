#!/bin/bash

#SBATCH --job-name=pre1    # Job name
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (process)
#SBATCH --mem=150G                        # Memory per node (e.g., 4 GB)
#SBATCH --time=40:00:00                # Time limit (hh:mm:ss)


#SBATCH --export=all
#SBATCH --output=output/opt_pre1.txt
#SBATCH --error=output/err_pre1.txt

python -u corr_axon_dend.py
