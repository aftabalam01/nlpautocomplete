#!/usr/bin/env bash
lang_array=( 'english' )
for lng in "${lang_array[@]}"
do
    echo "Start Grader for ${lng}"
    ./grader/grader.sh ~/submit/ ${lng} ${lng}
    echo "End Grader for ${lng}"
done