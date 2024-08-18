#!/bin/bash -l
#SBATCH -J blood_lbae_experiments
#SBATCH -A plgrbm4heo-cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --error="error.txt"
#SBATCH --output="output.txt"
#SBATCH -p plgrid

ml miniconda3
eval "$(conda shell.bash hook)"
conda activate pixellevel

srun python ~/pixel-level-image-analysis/src/scripts/blood/blood_lbae_experiments_pipeline.py