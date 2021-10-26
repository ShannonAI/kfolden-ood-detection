#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# kfolden_bert.sh

TIME_SIGN=2021.10.21
FILE_NAME=nss_20news_6s_kfolden_bert
REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection

MODEL_SCALE=base
BERT_DIR=/data/lixiaoya/models/bert_cased_large
DATA_DIR=/data/lixiaoya/datasets/kfolden/20news_6s

OUTPUT_DIR=/data/lixiaoya/outputs/kfolden/2021.10.25/nss_20news_6s_kfolden_bert_base_4_256_3e-5_linear_0.1_1_1_1.0_0.002_0.1

EVAL_BATCH_SIZE=1
MAX_LENGTH=384

DATA_NAME=20news_6s
NUM_LEAVE_OUT_LABEL=1

export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=2 python ${REPO_PATH}/task/evaluate_saved_ckpt.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--max_length ${MAX_LENGTH} \
--enable_leave_label_out \
--num_of_left_label ${NUM_LEAVE_OUT_LABEL} \
--pretrained_plm_model

