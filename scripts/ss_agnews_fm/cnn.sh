#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: cnn.sh

TIME_SIGN=2021.10.21
FILE_NAME=nss_20news_6s_cnn
REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

PRECISION=32
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25

OUTPUT_BASE_DIR=/data/lixiaoya/outputs/kfolden
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${TIME_SIGN}/${FILE_NAME}

mkdir -p ${OUTPUT_DIR}

DATA_NAME=agnews_fm

DATA_DIR=/data/lixiaoya/datasets/kfolden_ood_detection/20news_6s
VOCAB_FILE=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d_vocab_400002.txt
LOG_FILE=${OUTPUT_DIR}/train_log.txt
INIT_EMBEDDING=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d.npy
VOCAB_SIZE=400002
EMB_SIZE=300
MODEL=cnn

MAX_LEN=128
EPOCH=20
OPTIM=torch.adam
WEIGHT_DECAY=5e-4
LR=0.001
LR_SCHEDULER=linear
WARMUP=0.0
MAX_CKPT=20
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=12
DROPOUT=0.2
CLASSIFIER=mlp
ACTIVATE=relu
NUM_LABEL=20
HIDDEN_SIZE=100
POOLING=max_pool

NUM_KERNEL=6
KERNEL_SIZE="3;4;5;6;7;8"
CONV_STRIDE="1;2;2;2;3;3"

CUDA_VISIBLE_DEVICES=1 python ${REPO_PATH}/task/train_nn.py \
--gpus="1" \
--data_name ${DATA_NAME} \
--default_root ${OUTPUT_DIR} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--precision=${PRECISION} \
--data_dir ${DATA_DIR} \
--vocab_file ${VOCAB_FILE} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--max_length ${MAX_LEN} \
--pad_to_max_length \
--do_lower_case \
--optimizer ${OPTIM} \
--weight_decay ${WEIGHT_DECAY} \
--lr ${LR} \
--lr_scheduler ${LR_SCHEDULER} \
--warmup_proportion ${WARMUP} \
--max_keep_ckpt ${MAX_CKPT} \
--output_dir ${OUTPUT_DIR} \
--log_file ${LOG_FILE} \
--dropout ${DROPOUT} \
--classifier_type ${CLASSIFIER} \
--activate_func ${ACTIVATE} \
--padding_idx 0 \
--init_word_embedding ${INIT_EMBEDDING} \
--vocab_size ${VOCAB_SIZE} \
--embedding_size ${EMB_SIZE} \
--hidden_size ${HIDDEN_SIZE} \
--pooling_strategy ${POOLING} \
--model_type ${MODEL} \
--distributed_backend 'dp' \
--gradient_clip_val 1.0 \
--num_kernels ${NUM_KERNEL} \
--kernel_size ${KERNEL_SIZE} \
--conv_stride ${CONV_STRIDE} \
--max_epochs ${EPOCH}