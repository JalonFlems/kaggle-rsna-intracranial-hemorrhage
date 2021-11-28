predict_meta() {
    oof=$1
    test=$2
    name=$3
    python -u -m src.meta.trainer --inputs-test "${test}" --inputs-oof "${oof}" --output-name ${name} |tee ./meta/${name}.log
}

oof100="[\
    ['./model/model100/fold0_ep2_valid_tta5.pkl', './model/model100/fold0_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold1_ep2_valid_tta5.pkl', './model/model100/fold1_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold2_ep2_valid_tta5.pkl', './model/model100/fold2_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold3_ep2_valid_tta5.pkl', './model/model100/fold3_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold4_ep2_valid_tta5.pkl', './model/model100/fold4_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold5_ep2_valid_tta5.pkl', './model/model100/fold5_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold6_ep2_valid_tta5.pkl', './model/model100/fold6_ep3_valid_tta5.pkl'],\
    ['./model/model100/fold7_ep2_valid_tta5.pkl', './model/model100/fold7_ep3_valid_tta5.pkl'],\
]"
test100="[\
    ['./model/model100/fold0_ep2_test_tta5.pkl', './model/model100/fold0_ep3_test_tta5.pkl'],\
    ['./model/model100/fold1_ep2_test_tta5.pkl', './model/model100/fold1_ep3_test_tta5.pkl'],\
    ['./model/model100/fold2_ep2_test_tta5.pkl', './model/model100/fold2_ep3_test_tta5.pkl'],\
    ['./model/model100/fold4_ep2_test_tta5.pkl', './model/model100/fold4_ep3_test_tta5.pkl'],\
    ['./model/model100/fold5_ep2_test_tta5.pkl', './model/model100/fold5_ep3_test_tta5.pkl'],\
    ['./model/model100/fold6_ep2_test_tta5.pkl', './model/model100/fold6_ep3_test_tta5.pkl'],\
    ['./model/model100/fold7_ep2_test_tta5.pkl'],\
]"

oof200="[\
    ['./model/model200/fold0_ep2_valid_tta5.pkl', './model/model200/fold0_ep3_valid_tta5.pkl'],\
    ['./model/model200/fold1_ep2_valid_tta5.pkl', './model/model200/fold1_ep3_valid_tta5.pkl'],\
    ['./model/model200/fold3_ep2_valid_tta5.pkl', './model/model200/fold3_ep3_valid_tta5.pkl'],\
    ['./model/model200/fold4_ep2_valid_tta5.pkl', './model/model200/fold4_ep3_valid_tta5.pkl'],\
    ['./model/model200/fold5_ep2_valid_tta5.pkl', './model/model200/fold5_ep3_valid_tta5.pkl'],\
    ['./model/model200/fold6_ep2_valid_tta5.pkl', './model/model200/fold6_ep3_valid_tta5.pkl'],\
]"

test200="[\
    ['./model/model200/fold0_ep2_test_tta5.pkl', './model/model200/fold0_ep3_test_tta5.pkl'],\
    ['./model/model200/fold1_ep2_test_tta5.pkl', './model/model200/fold1_ep3_test_tta5.pkl'],\
    ['./model/model200/fold3_ep2_test_tta5.pkl', './model/model200/fold3_ep3_test_tta5.pkl'],\
    ['./model/model200/fold4_ep2_test_tta5.pkl', './model/model200/fold4_ep3_test_tta5.pkl'],\
    ['./model/model200/fold5_ep2_test_tta5.pkl', './model/model200/fold5_ep3_test_tta5.pkl'],\
    ['./model/model200/fold6_ep2_test_tta5.pkl', './model/model200/fold6_ep3_test_tta5.pkl'],\
]"


oof100="[${oof100}]"
test100="[${test100}]"

oof200="[${oof200}]"
test200="[${test200}]"

oof_100_200="[${oof100}, ${oof200}]"
test_100_200="[${test100}, ${test200}]"

#predict_meta "${oof100}" "${test100}" meta100
predict_meta  "${oof200}" "${test200}" meta200
predict_meta  "${oof_100_200}" "${test_100_200}" meta_100_200
