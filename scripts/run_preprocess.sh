#!/bin/bash
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=0-05:00
#              D-hh:mm
#SBATCH --output=%N-%j.out

VENV_DIR=/project/6054584/mfjalon/venv

module load cuda
source $VENV_DIR/bin/activate
 
cd ~/projects/def-lakahrs/mfjalon/kaggle-rsna-intracranial-hemorrhage

DATA_DIR=~/scratch/rsna-intracranial-hemorrhage-detection
CACHE_DIR=~/projects/def-lakahrs/mfjalon/kaggle-rsna-intracranial-hemorrhage/cache

# train
python -m src.preprocess.dicom_to_dataframe --input $DATA_DIR/stage_2_train.csv --output $CACHE_DIR/train_raw.pkl --imgdir $DATA_DIR/stage_2_train
python -m src.preprocess.create_dataset --input $CACHE_DIR/train_raw.pkl --output $CACHE_DIR/train.pkl --brain-diff 60
python -m src.preprocess.make_folds --input $CACHE_DIR/train.pkl --output $CACHE_DIR/train_folds8_seed300.pkl --n-fold 8 --seed 300

# test
python -m src.preprocess.dicom_to_dataframe --input $DATA_DIR/stage_2_sample_submission.csv --output $CACHE_DIR/test_raw.pkl --imgdir $DATA_DIR/stage_2_test
python -m src.preprocess.create_dataset --input $CACHE_DIR/test_raw.pkl --output $CACHE_DIR/test.pkl --brain-diff 60
python -m src.preprocess.make_folds --input $CACHE_DIR/test.pkl --output $CACHE_DIR/test_folds8_seed300.pkl --n-fold 8 --seed 300
