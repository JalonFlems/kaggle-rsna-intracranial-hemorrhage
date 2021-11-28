#python -m src.postprocess.make_submission --inputs "['./meta/meta100_lgb.pkl', './meta/meta100_cat.pkl', './meta/meta100_xgb.pkl']" --output ./submission/sub001.csv
python -m src.postprocess.make_submission --inputs "['./meta/meta200_lgb.pkl', './meta/meta200_cat.pkl', './meta/meta200_xgb.pkl']" --output ./submission/sub002.csv
#kaggle competitions submit rsna-intracranial-hemorrhage-detection -m "" -f ./submission/sub001.csv
