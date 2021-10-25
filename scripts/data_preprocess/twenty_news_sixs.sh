#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: scripts/data_preprocess/twenty_news_sixs.sh

REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

TWENTY_NEWS_DATA=/data/lixiaoya/datasets/confidence/20news
SAVE_DATA_DIR=/data/lixiaoya/datasets/kfolden/20news_6s

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/non_semantic_shift/twenty_news_sixs.py \
--twentynews_data_dir ${TWENTY_NEWS_DATA} \
--save_data_dir ${SAVE_DATA_DIR}



