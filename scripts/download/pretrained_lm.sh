#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: pretrained_lm.sh

pip3 install tensorflow

SAVE_DIR=/data/lixiaoya/models

# for BERT-Uncased-Base
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip  -P ${SAVE_DIR}
unzip ${SAVE_DIR}/uncased_L-12_H-768_A-12.zip
rm ${SAVE_DIR}/uncased_L-12_H-768_A-12.zip

export BERT_UNCASED_BASE_DIR=${SAVE_DIR}/uncased_L-12_H-768_A-12
transformers-cli convert --model_type bert \
  --tf_checkpoint ${BERT_UNCASED_BASE_DIR}/bert_model.ckpt \
  --config ${BERT_UNCASED_BASE_DIR}/bert_config.json \
  --pytorch_dump_output ${BERT_UNCASED_BASE_DIR}/pytorch_model.bin
cp ${BERT_UNCASED_BASE_DIR}/bert_config.json ${BERT_UNCASED_BASE_DIR}/config.json
mv ${SAVE_DIR}/uncased_L-12_H-768_A-12  ${SAVE_DIR}/bert_uncased_base

# for BERT-Uncased-Large
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip -P ${SAVE_DIR}
unzip ${SAVE_DIR}/uncased_L-24_H-1024_A-16.zip
rm ${SAVE_DIR}/uncased_L-24_H-1024_A-16.zip
#
export BERT_UNCASED_LARGE_DIR=${SAVE_DIR}/uncased_L-24_H-1024_A-16
transformers-cli convert --model_type bert \
  --tf_checkpoint ${BERT_UNCASED_LARGE_DIR}/bert_model.ckpt \
  --config ${BERT_UNCASED_LARGE_DIR}/bert_config.json \
  --pytorch_dump_output ${BERT_UNCASED_LARGE_DIR}/pytorch_model.bin
cp ${BERT_UNCASED_LARGE_DIR}/bert_config.json  ${BERT_UNCASED_LARGE_DIR}/config.json
mv ${SAVE_DIR}/uncased_L-24_H-1024_A-16 ${SAVE_DIR}/bert_uncased_large


pip3 uninstall tensorflow
# pytorch-lightning 0.9.0 requires tensorboard==2.2.0,
# but you have tensorboard 2.7.0 which is incompatible.
pip3 install tensorboard==2.2.0