#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

#SBATCH --mem=5G
#SBATCH --time=0-30:00
#              D-hh:mm
#SBATCH --output=%N-%j.out
#SBATCH --error=%N-%j.err

# run_predict.sh <1, 2, 3, 4>
# To make efficient use of resources, we split predictions into 4 parts.
# This allows us to submit multiple jobs simultaneously.

# run_predict.sh 5 
# For googlenet

VENV_DIR=/project/6054584/mfjalon/venv

LATEST_RUN=/project/6054584/mfjalon/scripts/latest_run_${1}.txt

echo "Begin datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
module load cuda
source $VENV_DIR/bin/activate

 
cd ~/projects/def-lakahrs/mfjalon/kaggle-rsna-intracranial-hemorrhage
./bin/predict_pt_${1}.sh
echo "End datetime:" >> $LATEST_RUN
date >> $LATEST_RUN
