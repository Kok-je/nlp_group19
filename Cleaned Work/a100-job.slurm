#!/bin/bash

#SBATCH --job-name=a100_predict_job

## Request for 1 a100-40 node (can try a100-80) but 
#SBATCH --gpus=1
#SBATCH --gpus=a100-40

## Use gpu-long otherwise soc time limit is 3 hours
#SBATCH --partition=gpu-long

#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

#SBATCH --output=bert_predict_job_%j.out
#SBATCH --error=bert_predict_job_%j.err

echo "Job is running on $(hostname), started at $(date)"

# Cfm GPU status
nvidia-smi

source ./venv/bin/activate
echo "Using Python version: $(python --version)"

echo -e "\n====> Running Python Script...\n"
#python train.py
python predict.py
#python phase1_distillation_from_paper.py
echo -e "\n====> Finished running.\n"
echo -e "\nJob completed at $(date)"

deactivate
