#!/bin/bash

#SBATCH --job-name=spatial_test2 
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=03-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=spatial2_opt.txt
#SBATCH --error=spatial2_err.txt

python -u toy2.py
