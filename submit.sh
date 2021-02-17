#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Omkar Agashe,omagashe\nAftab Alam,aftaba\nAbduselam Shaltu,ashaltu" > submit/team.txt

# submit requirements.txt
cp -r requirements.txt submit/requirements.txt

# train model
#python src/main.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/main.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
# do not copy python compiled cache files
cp -r ./src ./submit/src
rm -rf ./submit/src/data/
rm -rf ./submit/src/model/__pycache__

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
