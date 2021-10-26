#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# kfolden_roberta.sh

TIME_SIGN=2021.10.21
FILE_NAME=nss_20news_6s_kfolden_roberta
REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection

MODEL_SCALE=base
BERT_DIR=/data/lixiaoya/models/roberta-large
DATA_DIR=/data/lixiaoya/datasets/kfolden_ood_detection/20news_6s

TRAIN_BATCH_SIZE=36
EVAL_BATCH_SIZE=12
MAX_LENGTH=256

OPTIMIZER=torch.adam
LR_SCHEDULE=linear
LR=3e-5

DATA_NAME=yahoo_answers_fm

BERT_DROPOUT=0.1
ACC_GRAD=1
MAX_EPOCH=5
GRAD_CLIP=1.0
WEIGHT_DECAY=0.002
WARMUP_PROPORTION=0.1

NUM_LEAVE_OUT_LABEL=1
LAMBDA_LOSS=0.2

PRECISION=16
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"
OUTPUT_BASE_DIR=/data/lixiaoya/outputs/kfolden
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${TIME_SIGN}/${FILE_NAME}_${MODEL_SCALE}_${TRAIN_BATCH_SIZE}_${MAX_LENGTH}_${LR}_${LR_SCHEDULE}_${BERT_DROPOUT}_${ACC_GRAD}_${MAX_EPOCH}_${GRAD_CLIP}_${WEIGHT_DECAY}_${WARMUP_PROPORTION}_${LOSS_SIGN}

mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=4 python ${REPO_PATH}/task/finetune_plm.py \
--gpus="1" \
--data_name ${DATA_NAME} \
--precision=${PRECISION} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--max_length ${MAX_LENGTH} \
--optimizer ${OPTIMIZER} \
--data_dir ${DATA_DIR} \
--hidden_dropout_prob ${BERT_DROPOUT} \
--bert_config_dir ${BERT_DIR} \
--lr ${LR} \
--lr_scheduler ${LR_SCHEDULE} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR} \
--max_epochs ${MAX_EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--weight_decay ${WEIGHT_DECAY} \
--warmup_proportion ${WARMUP_PROPORTION} \
--enable_leave_label_out \
--num_of_left_label ${NUM_LEAVE_OUT_LABEL} \
--lambda_loss ${LAMBDA_LOSS}
