gpu=0
tta=5

predict_valid() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    valid=./model/${model}/fold${fold}_ep${ep}_valid_tta${tta}.pkl

    python -m src.cnn.main valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}

predict_test() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    test=./model/${model}/fold${fold}_ep${ep}_test_tta${tta}.pkl

    python -m src.cnn.main test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}

#predict_valid model100 4 2
#predict_valid model100 4 3
#predict_valid model100 5 2
#predict_valid model100 5 3
#predict_valid model100 6 2
#predict_valid model100 6 3
#predict_valid model100 7 2
#predict_valid model100 7 3

#predict_test model100 4 2
#predict_test model100 4 3
#predict_test model100 5 2
#predict_test model100 5 3
#predict_test model100 6 2
#predict_test model100 6 3
#predict_test model100 7 2
#predict_test model100 7 3

predict_test model200 4 2
predict_test model200 4 3
predict_test model200 5 2
predict_test model200 5 3
predict_test model200 6 2
predict_test model200 6 3
predict_test model200 7 2
predict_test model200 7 3

#predict_test model210 4 2
#predict_test model210 4 3
#predict_test model210 5 2
#predict_test model210 5 3
#predict_test model210 6 2
#predict_test model210 6 3
#predict_test model210 7 2
#predict_test model210 7 3

