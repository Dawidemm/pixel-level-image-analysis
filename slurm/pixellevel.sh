#!/bin/bash -l
#SBATCH -J RBM_for_pixel-level_multispectral_image_segmentation
#SBATCH -A plgrbm4heo-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --error="error.txt"
#SBATCH --output="output.txt"
#SBATCH -p plgrid

ml miniconda3
eval "$(conda shell.bash hook)"
conda activate pixellevel

## srun python ~/pixel-level-image-analysis/src/scripts/hyperspectral_blood_train_pipeline.py
## srun python ~/pixel-level-image-analysis/exp_13/blood_seg.py
srun python ~/pixel-level-image-analysis/exp_13/blood_clustering.py