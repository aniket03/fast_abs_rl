#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=05:30:00
#SBATCH --mem=18000
#SBATCH --job-name=tr_rl
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

# python pre_cross_review_buc_files.py
# python analyze_corr_difficulty_scores.py


python train_full_rl.py --path='./models/rl_dir_buc_c' --cross-rev-bucket='c' --abs_dir='./models/abstractor_dir_buc_c'  --ext_dir='./models/extractor_dir_buc_c' --batch 64

