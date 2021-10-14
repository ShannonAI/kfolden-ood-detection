#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: yahoo_agnews_five.sh

REPO_PATH=/data/xiaoya/workspace/kfolden_ood_detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

YAHOO_ANSWERS_DIR=/data/xiaoya/datasets/confidence/yahoo_answers/yahoo_answers_csv
AGCORPUS_FILE=/data/xiaoya/datasets/confidence/agnews/newsspace200.xml
SAVE_DATA_DIR=/data/xiaoya/datasets/kfolden_ood_detection/yahoo_agnews_five

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/non_semantic_shift/yahoo_agnews_five.py \
--yahoo_answers_data_dir ${YAHOO_ANSWERS_DIR} \
--ag_corpus_data_file ${AGCORPUS_FILE} \
--save_data_dir ${SAVE_DATA_DIR}


