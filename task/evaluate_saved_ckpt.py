#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: evaluate_saved_ckpt.py

import os
import sys
import csv
import torch
import logging
import argparse

from utils.random_seed import set_random_seed
set_random_seed(2333)
csv.field_size_limit(sys.maxsize)

from transformers import AutoTokenizer
from data.datasets.label_fields import get_labels
from data.datasets.nn_doc_processor import NNDocProcessor
from data.datasets.plm_doc_processor import PLMDocProcessor
from metrics.accuracy import compute_accuracy
from metrics.model_calibration import compute_fpr_tpr95, compute_auroc, compute_id_aupr, compute_ood_aupr, compute_minimum_ood_detection_error_rate
from modules.get_single_model_result import collect_single_model_pred_logits, collect_single_model_confidence_score
from modules.get_kfolden_model_result import collect_kfolden_pred_logits, collect_kfolden_confidence_score
from modules.get_ensemble_model_result import collect_ensemble_models_pred_logits, collect_ensemble_models_confidence_score

def get_parser():
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_config_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_name", type=str, default="20news_6s")
    parser.add_argument("--enable_leave_label_out", action="store_true", help="leave label out")
    parser.add_argument("--num_of_left_label", default=0, type=int, help="number of labels as ood data distribution")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--model_type", default="plm", type=str)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--pretrained_plm_model", action="store_true", )
    parser.add_argument("--loss_name", type=str, default="ce", help=" The name of the task to  train.")
    parser.add_argument("--data_prefix", type=str, default="test")
    parser.add_argument("--temperature_value", type=float, default=1000)
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")
    parser.add_argument("--model_scale", type=str, default="single", choices=["kfolden", "single", "ensemble"])
    parser.add_argument("--num_of_ensemble", type=int, default=0)
    return parser


def set_confidence_strategy():
    # msp is short for maximum softmax probability
    # temp_scaling is short for temperature scaling
    # Mahalanobis is short for mahalanobis distance
    # dropout is short for dropout
    return ["msp", "temp_scaling", "mahalanobis", "dropout", "dropout+mahalanobis" ]


def collect_label_lst_from_data_files(data_dir, label_lst, dataset_name, data_distribution, data_prefix, pretrian_model=True):
    if pretrian_model:
        data_processor = PLMDocProcessor(data_dir, dataset_name=dataset_name)
    else:
        data_processor = NNDocProcessor(data_dir, dataset_name=dataset_name)

    if data_prefix == "dev":
        data_examples = data_processor.get_dev_examples(dist_sign=data_distribution)
    elif data_prefix == "test":
        data_examples = data_processor.get_test_examples(dist_sign=data_distribution)
    else:
        raise ValueError

    if data_distribution == "id":
        collected_label_term_lst = [data_item["label"] for data_item in data_examples]
        label_map = {key: value for value, key in enumerate(label_lst)}
        collected_label_idx_lst = [label_map[item] for item in collected_label_term_lst]
        return collected_label_idx_lst, collected_label_term_lst
    elif data_distribution == "ood":
        collected_label_term_lst = [data_item["label"] for data_item in data_examples]
        return collected_label_term_lst
    else:
        raise ValueError("data_distribution not in collect_label_lst_from_data_files")


def evaluate_performances(id_pred_label_idx_array, id_dataset_label_idx_lst, id_confidence_array, ood_confidence_array, id_label_lst):
    num_id_labels = len(id_label_lst)
    id_acc = compute_accuracy(id_pred_label_idx_array, id_dataset_label_idx_lst)
    fpr_value = compute_fpr_tpr95(id_confidence_array, ood_confidence_array, num_id_labels)
    auroc_value = compute_auroc(id_confidence_array, ood_confidence_array, num_id_labels)
    id_aupr_value = compute_id_aupr(id_confidence_array, ood_confidence_array, num_id_labels)
    ood_aupr_value = compute_ood_aupr(id_confidence_array, ood_confidence_array, num_id_labels)
    mini_ood_error = compute_minimum_ood_detection_error_rate(id_confidence_array, ood_confidence_array, num_id_labels)

    return id_acc, fpr_value, auroc_value, id_aupr_value, ood_aupr_value, mini_ood_error


def main():
    # parse input arguments
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    format = '%(asctime)s - %(name)s - %(message)s'
    logging.basicConfig(format=format, filename=os.path.join(args.output_dir, "eval", "eval_result_log.txt"), level=logging.INFO)
    result_logger = logging.getLogger(__name__)
    result_logger.setLevel(logging.INFO)

    # use cuda or not
    use_cuda = torch.cuda.is_available()
    working_device = torch.device(f"cuda:{torch.cuda.current_device()}" if use_cuda else "cpu")
    result_logger.info(f">>> use cuda: {use_cuda}; working device: {working_device}")

    # load tokenizer
    full_id_label_lst = get_labels(data_sign=args.data_name, dist_sign="id")
    num_id_labels = len(full_id_label_lst)
    result_logger.info(f">>> TRAIN labels : {full_id_label_lst}")
    if args.pretrained_plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir, use_fast=False, do_lower_case=args.do_lower_case)
    else:
        tokenizer = None
    # start eval process
    save_eval_dir = os.path.join(args.output_dir, "eval",)

    if args.enable_leave_label_out:
        # prepare for in-distribution label list, need to turn to visible labels
        full_label_idx_lst, full_label_term_lst = collect_label_lst_from_data_files(args.data_dir, full_id_label_lst, args.data_name, "id", args.data_prefix, pretrian_model=args.pretrained_plm_model)
        collect_kfolden_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="id")
        full_ood_label_term_lst = collect_label_lst_from_data_files(args.data_dir, full_id_label_lst, args.data_name, "ood", args.data_prefix, pretrian_model=args.pretrained_plm_model)
        collect_kfolden_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="ood")

        confidence_strategy_lst = set_confidence_strategy()
        for strategy_item in confidence_strategy_lst:
            id_pred_label_idx_array, id_confidence_array, ood_confidence_array = collect_kfolden_confidence_score(args.num_of_left_label, save_eval_dir, full_id_label_lst, confidence_strategy=strategy_item)
            id_acc, fpr_value, auroc_value, id_aupr_value, ood_aupr_value, mini_ood_error = evaluate_performances(id_pred_label_idx_array, full_label_idx_lst, id_confidence_array,
                                                                                                                  ood_confidence_array, full_id_label_lst)

            result_logger.info("%%%"*60)
            result_logger.info(f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                               f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
            result_logger.info("%%%"*60)
    else:
        if args.model_scale == "single":
            full_label_idx_lst, full_label_term_lst = collect_label_lst_from_data_files(args.data_dir, full_id_label_lst, args.data_name, "id",
                                                                                        args.data_prefix, pretrian_model=args.pretrained_plm_model)
            collect_single_model_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="id")
            collect_single_model_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="ood")
            confidence_strategy_lst = set_confidence_strategy()
            for strategy_item in confidence_strategy_lst:
                id_pred_label_idx_array, id_confidence_array, ood_confidence_array = collect_single_model_confidence_score(save_eval_dir,
                                                                                                                           confidence_strategy=strategy_item,
                                                                                                                           temperature_value=args.temperature_value)
                id_acc, fpr_value, auroc_value, id_aupr_value, ood_aupr_value, mini_ood_error = evaluate_performances(id_pred_label_idx_array, full_label_idx_lst, id_confidence_array,
                                                                                                                      ood_confidence_array, full_id_label_lst)

                result_logger.info("%%%" * 60)
                result_logger.info(f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                                   f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
                result_logger.info("%%%" * 60)

        elif args.model_scale == "ensemble":
            full_label_idx_lst, full_label_term_lst = collect_label_lst_from_data_files(args.data_dir, full_id_label_lst, args.data_name, "id",
                                                                                        args.data_prefix, pretrian_model=args.pretrained_plm_model)
            if args.num_of_ensemble == 0:
                num_of_ensemble = len(full_id_label_lst)
            else:
                num_of_ensemble = args.num_of_ensemble
            collect_ensemble_models_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, num_of_ensemble, dist_sign="id")
            collect_ensemble_models_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, num_of_ensemble, dist_sign="ood")
            confidence_strategy_lst = set_confidence_strategy()
            for strategy_item in confidence_strategy_lst:
                id_pred_label_idx_array, id_confidence_array, ood_confidence_array = collect_ensemble_models_confidence_score(save_eval_dir, confidence_strategy=strategy_item,
                                                                                                                              temperature_value=args.temperature_value)
                id_acc, fpr_value, auroc_value, id_aupr_value, ood_aupr_value, mini_ood_error = evaluate_performances(id_pred_label_idx_array, full_label_idx_lst, id_confidence_array,
                                                                                                                      ood_confidence_array, full_id_label_lst)

                result_logger.info("%%%" * 60)
                result_logger.info(f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                                   f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
                result_logger.info("%%%" * 60)
        else:
            raise ValueError("Please append --enable_leave_label_out in you command . ")


if __name__ == "__main__":
    main()