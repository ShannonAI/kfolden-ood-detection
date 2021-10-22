#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# kfolden_bert.sh

TIME_SIGN=2021.10.21
FILE_NAME=nss_20news_6s_kfolden_bert
REPO_PATH=/data/xiaoya/workspace/kfolden-ood-detection

MODEL_SCALE=base
BERT_DIR=/data/xiaoya/pretrain_lm/cased_L-12_H-768_A-12
DATA_DIR=/data/xiaoya/datasets/kfolden_ood_detection/20news_6s

EVAL_BATCH_SIZE=12
MAX_LENGTH=64

NUM_LEAVE_OUT_LABEL=1

export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"
OUTPUT_BASE_DIR=/data/lixiaoya/outputs/kfolden
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${TIME_SIGN}/${FILE_NAME}_${MODEL_SCALE}_${TRAIN_BATCH_SIZE}_${MAX_LENGTH}_${LR}_${LR_SCHEDULE}_${BERT_DROPOUT}_${ACC_GRAD}_${MAX_EPOCH}_${GRAD_CLIP}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${LOSS_SIGN}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=3 python ${REPO_PATH}/task/evaluate_plm_model.py \
--gpus="1" \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--max_length ${MAX_LENGTH} \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--default_root_dir ${OUTPUT_DIR} \
--enable_leave_label_out \
--num_of_left_label ${NUM_LEAVE_OUT_LABEL} \
--loss_name kfolden

