#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: kfolden_lstm.sh

TIME_SIGN=2021.10.21
SCRIPT_SIGN=kfolden
FILE_NAME=nss_agnews_ext_${SCRIPT_SIGN}_lstm
REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection

LOSS_NAME=kfolden
DATA_NAME=agnews_ext
MODEL_SCALE=kfolden
MODEL_TYPE=rnn

PRECISION=32
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

OUTPUT_BASE_DIR=/data/lixiaoya/outputs/kfolden
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${TIME_SIGN}/${FILE_NAME}

mkdir -p ${OUTPUT_DIR}

DATA_DIR=/data/lixiaoya/datasets/kfolden/agnews_ext
VOCAB_FILE=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d_vocab_400002.txt
INIT_EMBEDDING=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d.npy
VOCAB_SIZE=400002
EMB_SIZE=300
GRAD_CLIP=1.0

PAD_IDX=0
MAX_LEN=384
EPOCH=20
OPTIM=torch.adam
WEIGHT_DECAY=5e-4
LR=0.003
LR_SCHEDULER=polydecay
WARMUP=0.0
MAX_CKPT=20
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=12
DROPOUT=0.1
CLASSIFIER=mlp
ACTIVATE=relu
NUM_LABEL=20
HIDDEN_SIZE=300
NUM_LAYER=1
RNN_DROPOUT=0.2
RNN_ACT=tanh
POOLING=max_pool

CUDA_VISIBLE_DEVICES=5 python ${REPO_PATH}/task/train_nn.py \
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
--dropout ${DROPOUT} \
--classifier_type ${CLASSIFIER} \
--activate_func ${ACTIVATE} \
--padding_idx ${PAD_IDX} \
--init_word_embedding ${INIT_EMBEDDING} \
--vocab_size ${VOCAB_SIZE} \
--embedding_size ${EMB_SIZE} \
--hidden_size ${HIDDEN_SIZE} \
--num_layers ${NUM_LAYER} \
--rnn_dropout ${RNN_DROPOUT} \
--rnn_activate_func ${RNN_ACT} \
--pooling_strategy ${POOLING} \
--model_type ${MODEL_TYPE} \
--max_epochs ${EPOCH} \
--gradient_clip_val ${GRAD_CLIP} \
--bidirectional \
--rnn_cell_type lstm \
--distributed_backend 'dp' \
--loss_name ${LOSS_NAME} \
--model_scale ${MODEL_SCALE}