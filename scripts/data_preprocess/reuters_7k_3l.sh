#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: scripts/data_preprocess/reuters_7k_3l.sh

REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

AGNEWS_DATA=/data/lixiaoya/datasets/confidence/agnews/ag_news_csv
AGCORPUS_FILE=/data/lixiaoya/datasets/confidence/agnews/newsspace200.xml
SAVE_DATA_DIR=/data/lixiaoya/datasets/kfolden/agnews_ext

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/semantic_shift/reuters_7k_3l.py \
--ag_news_data_dir ${AGNEWS_DATA} \
--ag_corpus_data_file ${AGCORPUS_FILE} \
--save_data_dir ${SAVE_DATA_DIR}

