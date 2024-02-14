#!/bin/bash

#SBATCH --job-name=23Ppost    # Job name
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought             
#SBATCH --partition=compute
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task (process)
#SBATCH --mem=256G                        # Memory per node (e.g., 4 GB)
#SBATCH --time=96:00:00                # Time limit (hh:mm:ss)


#SBATCH --export=all
#SBATCH --output=output/opt_23Ppost.txt
#SBATCH --error=output/err_23Ppost.txt

python -u corr_axon_dend.py
