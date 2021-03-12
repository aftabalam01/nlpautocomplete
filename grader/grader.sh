#!/usr/bin/env bash
# usage make sure input and answer file in prefix with $3
# /grader.sh ~/submit/ english english
set -e
set -x

DATA=$1
OUT=${2:-output}
LANG=$3
mkdir -p $OUT

docker build -t cse447-proj/demo -f Dockerfile .

function run() {
  docker run --rm \
    -v $PWD/src:/job/src \
    -v $PWD/work:/job/work \
    -v $DATA:/job/data \
    -v $PWD/$OUT:/job/output \
    cse447-proj/demo \
    bash /job/src/predict.sh /job/data/${LANG}_input.txt /job/output/pred.txt
}

(time run) > $OUT/output 2>$OUT/runtime
python3 grader/grade.py $OUT/pred.txt $DATA/${LANG}_answer.txt > $OUT/success
