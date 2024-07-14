#!/bin/bash -l
#SBATCH -J RBM for pixel-level multispectral image segmentation
#SBATCH -A plgrbm4heo-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH -p plgrid

source ~/.conda/envs/pixellevel/bin/activate
srun python ~/pixel-level-image-analysis/src/scripts/hyperspectral_blood_train_pipeline.py