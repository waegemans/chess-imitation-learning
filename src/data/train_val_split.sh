#!/bin/bash
grep -e ' b ' -v moves.csv | shuf > moves_shuf.csv
val=$(bc <<< $(cat moves_shuf.csv | wc -l)*0.1/1)
head -n -$val moves_shuf.csv > moves_train.csv
tail -n -$val moves_shuf.csv > moves_val.csv
