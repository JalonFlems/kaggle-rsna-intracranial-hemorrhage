gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}

train model200 0
train model200 1
train model200 2
train model200 3
train model200 4
train model200 5
train model200 6
train model200 7







