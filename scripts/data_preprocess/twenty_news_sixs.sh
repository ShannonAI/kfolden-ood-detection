#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: scripts/data_preprocess/twenty_news_sixs.sh

REPO_PATH=/data/xiaoya/workspace/kfolden_ood_detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

TWENTY_NEWS_DATA=/data/xiaoya/datasets/confidence/20news
SAVE_DATA_DIR=/data/xiaoya/datasets/kfolden_ood_detection/20news_6s

mkdir -p ${SAVE_DATA_DIR}

python3 ${REPO_PATH}/data/preprocess/non_semantic_shift/twenty_news_sixs.py \
--twentynews_data_dir ${TWENTY_NEWS_DATA} \
--save_data_dir ${SAVE_DATA_DIR}



