#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: agnews_fl.sh

REPO_PATH=/data/xiaoya/workspace/kfolden_ood_detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

AGNEWS_DATA_DIR=/data/xiaoya/datasets/confidence/agnews/ag_news_csv
AGCORPUS_FILE=/data/xiaoya/datasets/confidence/agnews/newsspace200.xml
SAVE_DATA_DIR=/data/xiaoya/datasets/kfolden_ood_detection/agnews_fl

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/semantic_shift/agnews_fl.py \
--ag_news_data_dir ${AGNEWS_DATA_DIR} \
--ag_corpus_data_file ${AGCORPUS_FILE} \
--save_data_dir ${SAVE_DATA_DIR}
