#!/usr/bin/env bash
set -eu

FONT_DIRPATH="./ttfs"
DST_DIRPATH="datasets/fonts/english_ttf"

#CN_MODEL_PATH="PretrainCelebA/model_cn"

python copy_font.py \
        ${FONT_DIRPATH} \
        ${DST_DIRPATH} \