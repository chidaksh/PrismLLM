#!/bin/bash

# Parameters
#SBATCH --error=%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_inf
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --output=%A_%a_0_log.out
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --time=8639
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=chidaksh@ad.unc.edu

module load cuda/12.2
module load anaconda/2023.03
conda activate hack
python llama.py