#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --mem=5G
#SBATCH --time=0-30:00
#              D-hh:mm
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

# Ensembles all model weights 

VENV_DIR=/project/6054584/mfjalon/venv

LATEST_RUN=/project/6054584/mfjalon/scripts/latest_ensemble.txt

echo "Begin datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
module load cuda
source $VENV_DIR/bin/activate

 
cd ~/projects/def-lakahrs/mfjalon/kaggle-rsna-intracranial-hemorrhage
./bin/ensemble.sh
echo "End datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
