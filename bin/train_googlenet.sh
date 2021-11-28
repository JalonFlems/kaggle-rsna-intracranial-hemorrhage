gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python -m src.cnn.main train ${conf} --fold ${fold} --gpu ${gpu}
}


train model210 0
train model210 1
train model210 2
train model210 3
train model210 4
train model210 5
train model210 6
train model210 7




