#!/bin/bash

# Logging configuration
#SBATCH --job-name=data
#SBATCH --output=/private/home/vivc/logs_data/stdout/%j/main.out
#SBATCH --error=/private/home/vivc/logs_data/stdout/%j/main.err
#SBATCH --open-mode=append

# Job specification
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1440

python -m apps.llm.scripts.download_dataset hf-download --name dclm