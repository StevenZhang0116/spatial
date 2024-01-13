#!/bin/bash

#SBATCH --job-name=33
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=8G            
#SBATCH --time=03-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=spatial_opt33.txt
#SBATCH --error=spatial_err33.txt

python -u toy.py