#!/bin/bash

#Data setup script for SRNet
#Author: Niwhskal
#https://github.com/Niwhskal/

echo "setting up dirs ($(date))"

main_dir=$(pwd)'/'
echo "This is $main_dir"

cd "$main_dir"

mkdir -p "datasets/"
cd "datasets"

mkdir -p "csnet_data/"
mkdir -p "fonts/english_ttf/"

mkdir -p "bg_data"


if [ -f imnames.cp ]; then
  :
else
  echo "Downloading background image names ($(date))"
  wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/imnames.cp 
fi

echo "----------"

if [ -e ./bg_data/bg_img.tar.gz ]; then
  :
else
  echo "Downloading background images ($(date))"
  cd "bg_data"
  wget http://www.robots.ox.ac.uk/~vgg/data/scenetext/preproc/bg_img.tar.gz
  tar -xzf bg_img.tar.gz
fi

echo "Changing Paths $(date)"

copy_path="${main_dir}datasets/"
code_path="${main_dir}SRNet-Datagen/"

cd "${code_path}"

sed -i "s|^data_dir.*$|data_dir = '${copy_path}csnet_data/'|" cfg.py

cd "Synthtext"

sed -i "s|^font_dir.*$|font_dir = '${copy_path}fonts/english_ttf/'|" data_cfg.py

sed -i "s|^standard_font_path.*$|standard_font_path = '${copy_path}fonts/english_ttf/arial.ttf'|" data_cfg.py

sed -i "s|^bg_filepath.*$|bg_filepath = '${copy_path}imnames.cp'|" data_cfg.py

sed -i "s|^temp_bg_path.*$|temp_bg_path = '${copy_path}bg_data/bg_img/'|" data_cfg.py

echo "Moving fonts ($(date))"

cd ${main_dir}datasets/fonts/english_ttf
cp ${code_path}arial.ttf ./
cp ${code_path}OpenSans-Regular.ttf ./

cd "${code_path}"

python3 datagen.py

echo "COMPLETED AT $(date)"