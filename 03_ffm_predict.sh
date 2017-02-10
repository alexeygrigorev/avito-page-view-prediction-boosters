# FFM regression version taken from 
# https://github.com/bobye/libffm-regression/tree/master/alpha-regression
#
# script assumes FFM is built and the binaries on PATH

cd ffm

PARAMS='-s 8 -k 32 -l 0.0001 -t 70 -r 0.01'

ffm-train $PARAMS -p ffm1_time.txt ffm0_time.txt ffm1_time_model.bin
ffm-train $PARAMS -p ffm0_time.txt ffm1_time.txt ffm0_time_model.bin

cat ffm0_time.txt ffm1_time.txt > ffm_full_time.txt
ffm-train $PARAMS ffm_full_time.txt ffm_full_time_model.bin

ffm-predict ffm1_time.txt ffm1_time_model.bin pred_1_time.txt
ffm-predict ffm0_time.txt ffm0_time_model.bin pred_0_time.txt
ffm-predict ffm_test_time.txt ffm_full_time_model.bin pred_test_time.txt