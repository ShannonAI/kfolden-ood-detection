#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: word_embedding.sh

REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"
SAVE_EMB_DIR=/data/lixiaoya/models/embeddings

wget https://nlp.stanford.edu/data/glove.840B.300d.zip -P ${SAVE_EMB_DIR}
unzip ${SAVE_EMB_DIR}/glove.840B.300d.zip
rm ${SAVE_EMB_DIR}/glove.840B.300d.zip

EMB_FILE=${SAVE_EMB_DIR}/glove.840B.300d.txt

python3 ${REPO_PATH}/data/data_utils/transform_word_emb.py ${EMB_FILE} ${SAVE_EMB_DIR}
