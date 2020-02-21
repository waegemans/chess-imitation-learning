#!/bin/bash
grep -e ' w ' -v moves_statevalue.csv | shuf > moves_statevalue_shuf.csv
val=$(bc <<< $(cat moves_statevalue_shuf.csv | wc -l)*0.1/1)
head -n -$val moves_statevalue_shuf.csv > moves_statevalue_train.csv
tail -n -$val moves_statevalue_shuf.csv > moves_statevalue_val.csv