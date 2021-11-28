#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --mem=150G
#SBATCH --time=0-48:00
#              D-hh:mm
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

VENV_DIR=/project/6054584/mfjalon/venv

LATEST_RUN=/project/6054584/mfjalon/scripts/latest_train_results_sp${2}_${1}.txt

echo "Begin datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
module load cuda
source $VENV_DIR/bin/activate

cd ~/projects/def-lakahrs/mfjalon/kaggle-rsna-intracranial-hemorrhage
./bin/train_new_model_sp${2}_${1}.sh
echo "End datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
