#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: get_parser.py

import argparse

def get_plm_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = argparse.ArgumentParser(description="argument parser")

    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--data_dir", type=str, help="data dir")
    parser.add_argument("--bert_config_dir", type=str, help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--train_batch_size", type=int, default=32, help="batch size for train dataloader")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for eval dataloader")
    parser.add_argument("--lambda_loss", type=float, default=0.5, )
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", help="type of lr scheduler")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--enable_leave_label_out", action="store_true", help="leave label out")
    parser.add_argument("--num_of_left_label", default=0, type=int, help="number of labels as ood data distribution")
    # number of data-loader workers should equal to 0.
    # https://blog.csdn.net/breeze210/article/details/99679048
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    # in case of not error, define a new argument
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_keep_ckpt", default=3, type=int,
                        help="the number of keeping ckpt max.")
    parser.add_argument("--output_dir", default="/data", type=str, help="the directory to save model outputs")
    parser.add_argument("--only_keep_the_best_ckpt_after_training", action="store_true", help="only the best model checkpoint after training. ")
    return parser

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--data_dir", type=str, help="data dir")
    parser.add_argument("--allow_ood_data", action="store_true")
    parser.add_argument("--label_file", type=str, default="benchmark_label.txt")
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")
    parser.add_argument("--train_batch_size", type=int, default=32, help="batch size for train dataloader")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for eval dataloader")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--max_length", type=int, default=128,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequence longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pad_to_max_length", action="store_false", help="Whether to pad all samples to ' max_seq_length'.")
    # number of data-loader workers should equal to 0.
    # https://blog.csdn.net/breeze210/article/details/99679048
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--enable_leave_label_out", action="store_true", help="leave label out")
    parser.add_argument("--num_of_left_label", default=0, type=int, help="number of labels as ood data distribution")

    parser.add_argument("--optimizer", default="adamw", help="loss type")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="onecycle", help="type of lr scheduler")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_keep_ckpt", default=20, type=int, help="the number of keeping ckpt max.")
    parser.add_argument("--output_dir", default="/data", type=str, help="the directory to save model outputs")
    parser.add_argument("--log_file", default="train_log.txt", type=str, )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
    # only keep the best checkpoint after training.
    parser.add_argument("--only_keep_the_best_ckpt_after_training", action="store_true", help="only the best model checkpoint after training. ")

    return parser

def add_bert_configurations(base_parser: argparse.ArgumentParser):
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, )
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--bert_config_dir", type=str, help="bert config dir")
    parser.add_argument("--truncated_normal", action="store_true")

    return parser

def add_basic_configurations(base_parser: argparse.ArgumentParser):
    # vanilla model params
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    parser.add_argument("--classifier_type", type=str, default="mlp")
    parser.add_argument("--activate_func", type=str, default="gelu")
    parser.add_argument("--padding_idx", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--init_word_embedding", type=str, default="")
    parser.add_argument("--freeze_word_embedding", action="store_true", )
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--pooling_strategy", type=str, default="avg_pool")

    return parser

def add_rnn_configurations(base_parser: argparse.ArgumentParser):
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)

    # model specific config
    parser.add_argument("--batch_first", action="store_false")
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--rnn_activate_func", type=str, default="relu")
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--rnn_cell_type", type=str, default="rnn", choices=["rnn", "lstm", "gru"])
    return parser

def add_cnn_configurations(base_parser: argparse.ArgumentParser):
    # vanilla model params
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)

    # cnn specific
    parser.add_argument("--num_kernels", type=int, default=3)
    parser.add_argument("--kernel_size", type=str, default="3;4;5")
    parser.add_argument("--conv_stride", type=str, default="3;4;5")

    return parser