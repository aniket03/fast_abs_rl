#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=06:00:00
#SBATCH --mem=18000
#SBATCH --job-name=tr_both
#SBATCH --mail-type=END
#SBATCH --mail-user=ab8700@nyu.edu
#SBATCH --output=slurm_%j.out

python train_extractor_ml.py  --path='./models/extractor_dir_buc_3' --cross-rev-bucket='3' --w2v='./models/word2vec_dir/word2vec.128d.198k.bin' --batch 64
python train_abstractor.py  --path='./models/abstractor_dir_buc_3' --cross-rev-bucket='3' --w2v='./models/word2vec_dir/word2vec.128d.198k.bin' --batch 64
python train_full_rl.py --path='./models/rl_dir_buc_3' --cross-rev-bucket='3' --abs_dir='./models/abstractor_dir_buc_3'  --ext_dir='./models/extractor_dir_buc_3' --batch 64
