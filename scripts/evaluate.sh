#!/bin/bash


#SBATCH --account=nlprx-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --output=local_search.out
#SBATCH --constraint=a40

source /srv/share5/emendes3/miniconda3/etc/profile.d/conda.sh
conda activate bertenv3.8.1
python3 /nethome/emendes3/Complementary-Performance/train/train_stance_saliency.py
conda deactivate