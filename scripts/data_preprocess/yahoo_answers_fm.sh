#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: yahoo_answers_fm.sh

REPO_PATH=/data/xiaoya/workspace/kfolden_ood_detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"


YAHOO_ANSWERS_DIR=/data/xiaoya/datasets/confidence/yahoo_answers/yahoo_answers_csv
SAVE_DATA_DIR=/data/xiaoya/datasets/kfolden_ood_detection/yahoo_answers_fm

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/semantic_shift/yahoo_answers_fm.py \
--yahoo_answers_data_dir ${YAHOO_ANSWERS_DIR} \
--save_data_dir ${SAVE_DATA_DIR}