#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0

echo "Beginning training script for foot"
python ../train.py --config_path ../configs/default.yml --dataset gestational_foot


