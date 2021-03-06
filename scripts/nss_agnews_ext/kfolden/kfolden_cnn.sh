#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: cnn.sh

TIME_SIGN=2021.10.29
SCRIPT_SIGN=kfolden
FILE_NAME=nss_agnews_ext_${SCRIPT_SIGN}_cnn
REPO_PATH=/data/lixiaoya/workspace/kfolden-ood-detection
export PYTHONPATH="$PYTHONPATH:${REPO_PATH}"

LOSS_NAME=kfolden
DATA_NAME=agnews_ext
MODEL_SCALE=single
MODEL_TYPE=cnn

NUM_LEAVE_OUT_LABEL=1
LAMBDA_LOSS=0.001

PRECISION=32
PROGRESS_BAR=1
VAL_CHECK_INTERVAL=0.25

OUTPUT_BASE_DIR=/data/lixiaoya/outputs/kfolden_outputs
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${TIME_SIGN}/${FILE_NAME}

mkdir -p ${OUTPUT_DIR}

DATA_DIR=/data/lixiaoya/datasets/kfolden/agnews_ext
VOCAB_FILE=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d_vocab_400002.txt
INIT_EMBEDDING=/data/lixiaoya/datasets/confidence/embeddings/glove.6B.300d.npy
VOCAB_SIZE=400002
EMB_SIZE=300

MAX_LEN=384
PAD_IDX=0
EPOCH=20
OPTIM=torch.adam
WEIGHT_DECAY=1e-4
LR=0.001
GRAD_CLIP=1.0
LR_SCHEDULER=linear
WARMUP=0.0
MAX_CKPT=5
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=1
DROPOUT=0.2
CLASSIFIER=mlp
ACTIVATE=relu
HIDDEN_SIZE=100
POOLING=max_pool

NUM_KERNEL=6
KERNEL_SIZE="3;4;5;6;7;8"
CONV_STRIDE="1;2;2;2;3;3"

CUDA_VISIBLE_DEVICES=3 python ${REPO_PATH}/task/train_nn.py \
--gpus="1" \
--default_root ${OUTPUT_DIR} \
--data_name ${DATA_NAME} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--precision=${PRECISION} \
--data_dir ${DATA_DIR} \
--vocab_file ${VOCAB_FILE} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--eval_batch_size ${EVAL_BATCH_SIZE} \
--max_length ${MAX_LEN} \
--pad_to_max_length \
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
--pooling_strategy ${POOLING} \
--model_type ${MODEL_TYPE} \
--distributed_backend 'dp' \
--gradient_clip_val ${GRAD_CLIP} \
--num_kernels ${NUM_KERNEL} \
--kernel_size ${KERNEL_SIZE} \
--conv_stride ${CONV_STRIDE} \
--max_epochs ${EPOCH} \
--loss_name ${LOSS_NAME} \
--model_scale ${MODEL_SCALE} \
--do_lower_case \
--enable_leave_label_out \
--num_of_left_label ${NUM_LEAVE_OUT_LABEL} \
--lambda_loss ${LAMBDA_LOSS} \
--loss_name ${LOSS_NAME} \
--model_scale ${MODEL_SCALE}
