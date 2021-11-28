gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}

train model100_exp 0
train model100_exp 1
train model100_exp 2
train model100_exp 3
train model100_exp 4
train model100_exp 5
train model100_exp 6
train model100_exp 7

#train model210 0
#train model210 1
#train model210 2
#train model210 3
#train model210 4
#train model210 5
#train model210 6
#train model210 7
