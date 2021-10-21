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


predict_valid model100 0 2
