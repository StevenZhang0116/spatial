#!/bin/bash
#SBATCH --job-name=my_cpp_job             # Job name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks-per-node=1               # Number of tasks (processes) per node
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task (process)
#SBATCH --mem=900G                        # Memory per node (e.g., 4 GB)
#SBATCH --time=144:00:00                   # Time limit (hh:mm:ss)

#SBATCH --export=all
#SBATCH --output=opt.txt
#SBATCH --error=err.txt

python -u toy.py