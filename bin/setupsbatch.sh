#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --mem=4G
#SBATCH --time=0-00:10
#SBATCH --output=%N-%j.out

date > latest_run.txt

module load python/3.8


source ENV/bin/activate
pip install joblib --no-index


