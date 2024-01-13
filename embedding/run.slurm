#!/bin/bash

#SBATCH --job-name=run_test  
#SBATCH --mail-user=zihan16@uw.edu

#SBATCH --account=deepthought
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1           
#SBATCH --mem=64G            
#SBATCH --time=04-12:00:00      
#SBATCH --gpus=1

#SBATCH --export=all
#SBATCH --output=run_opt.txt
#SBATCH --error=run_err.txt

/gscratch/deepthought/zihan/miniconda3/envs/embedding/bin/python -u seq2seq_translation.py
