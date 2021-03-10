#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-10.0
module load nvidia/cudnn-v7.6.5.32-forcuda10.0

#export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
#export PATH="/usr/local/cuda/bin:$PATH"

echo "Beginning training script for face"
python ../train.py --config_path ../configs/default.yml --dataset gestational_face

echo "Anime test"
python ../train.py --config_path ../configs/default.yml

