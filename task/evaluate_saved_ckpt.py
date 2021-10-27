#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: evaluate_saved_ckpt.py

import os
import sys
import csv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

from utils.random_seed import set_random_seed
set_random_seed(2333)
csv.field_size_limit(sys.maxsize)

from transformers import AutoTokenizer
from data.datasets.label_fields import get_labels
from data.datasets.nn_doc_dataset import NNDocDataset
from data.datasets.plm_doc_dataset import PLMDocDataset
from data.datasets.nn_doc_processor import NNDocProcessor
from data.datasets.plm_doc_processor import PLMDocProcessor
from data.datasets.collate_functions import collate_plm_to_max_length, collate_nn_to_max_length
from task.train_nn import TrainNNTask
from task.finetune_plm import FinetunePLMTask
from modules.pred_calibration import get_confidence_via_max_softmax_prob, get_confidence_via_temperature_scale, compute_softmax_np
from metrics.accuracy import compute_accuracy
from metrics.model_calibration import compute_fpr_tpr95, compute_auroc, compute_id_aupr, compute_ood_aupr, compute_minimum_ood_detection_error_rate


def get_parser():
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_config_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--data_name", type=str, default="20news_6s")
    parser.add_argument("--enable_leave_label_out", action="store_true", help="leave label out")
    parser.add_argument("--num_of_left_label", default=0, type=int, help="number of labels as ood data distribution")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--pretrained_plm_model", action="store_true", )
    parser.add_argument("--loss_name", type=str, default="ce", help=" The name of the task to  train.")
    parser.add_argument("--data_prefix", type=str, default="test")
    parser.add_argument("--temperature_value", type=float, default=1)
    parser.add_argument("--vocab_file", type=str, default="vocab.txt")
    parser.add_argument("--model_scale", type=str, default="single", choices=["kfolden", "single", "ensemble"])
    parser.add_argument("--num_of_ensemble", type=int, default=0)
    return parser


def set_confidence_strategy():
    # msp is short for maximum softmax probability
    # temp_scaling is short for temperature scaling
    # Mahalanobis is short for mahalanobis distance
    # dropout is short for dropout
    return ["msp", "temp_scaling", "mahalanobis", ""]

def get_dataloader(input_arguments, tokenizer, mode, keep_label_lst, pretrian_model=True, dist_sign="id"):
    batch_size = input_arguments.eval_batch_size
    if pretrian_model:
        dataset = PLMDocDataset(input_arguments, tokenizer, distribution_type=dist_sign, mode=mode, keep_label_lst=keep_label_lst)
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_plm_to_max_length)
    else:
        dataset = NNDocDataset(input_arguments, max_seq_length=input_arguments.max_length, keep_label_lst=keep_label_lst,
                               vocab_file=input_arguments.vocab_file, distribution_type=dist_sign, do_lower_case=input_arguments.do_lower_case)
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_nn_to_max_length)

    return dataloader


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
        raise ValueError


def collect_kfolden_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="id"):
    for idx in range(0, len(full_id_label_lst), args.num_of_left_label):
        print("$" * 60)
        print(f">>> Loading {idx} Model ...")
        left_label_lst = [item for item_idx, item in enumerate(full_id_label_lst) if item_idx in range(idx, idx + args.num_of_left_label)]
        keep_label_lst = [item for item_idx, item in enumerate(full_id_label_lst) if
                          item_idx not in range(idx, idx + args.num_of_left_label)]
        sub_output_dir = os.path.join(args.output_dir, f"{idx}")

        file_saving_best_ckpt_path = os.path.join(sub_output_dir, "best_ckpt_on_dev.txt")
        with open(file_saving_best_ckpt_path, "r") as f:
            best_ckpt_on_dev = f.read().strip()

        hparams_file = os.path.join(args.output_dir, "lightning_logs", "version_0", "hparams.yaml")
        model_ckpt = best_ckpt_on_dev
        print(f">>> load model checkpoint: {model_ckpt}")
        trained_model = FinetunePLMTask.load_from_checkpoint(keep_label_lst=keep_label_lst, checkpoint_path=model_ckpt, hparams_file=hparams_file,
                                                             map_location=None, batch_size=1, max_length=args.max_length, workers=0, )
        dataloader = get_dataloader(args, tokenizer, args.data_prefix, keep_label_lst, pretrian_model=args.pretrained_plm_model, dist_sign=dist_sign)
        pred_prob_results = []
        trained_model.model.to(working_device).eval()
        for batch in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                input_ids, token_type_ids, attention_mask = input_ids.to(working_device), token_type_ids.to(
                    working_device), attention_mask.to(working_device)
                output_logits = trained_model.model(input_ids, token_type_ids, attention_mask)
                output_probs = F.softmax(output_logits, dim=-1).detach().cpu().numpy()
                pred_prob_results.append(output_probs)
        pred_prob_tensor = np.squeeze(np.stack(pred_prob_results, axis=0))
        num_data = pred_prob_tensor.shape[0]
        pad_pred_prob = np.insert(pred_prob_tensor, idx, [[0]], axis=1)
        print(f">>> check the shape of pred_prob {pad_pred_prob.shape}")
        os.makedirs(save_eval_dir, exist_ok=True)
        os.system(f"chmod -R 777 {save_eval_dir}")
        path_to_save_np_results = os.path.join(save_eval_dir, f"{dist_sign}_{idx}.npy")
        np.save(path_to_save_np_results, pad_pred_prob)
        print(f">>> save pred probabilities to {path_to_save_np_results}")


def collect_ensemble_models_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, num_ensemble, dist_sign="id"):
    for idx in range(num_ensemble):
        print("$" * 60)
        print(f">>> Loading {idx} Model ...")
        sub_output_dir = os.path.join(args.output_dir, f"{idx}")

        file_saving_best_ckpt_path = os.path.join(sub_output_dir, "best_ckpt_on_dev.txt")
        with open(file_saving_best_ckpt_path, "r") as f:
            best_ckpt_on_dev = f.read().strip()

        hparams_file = os.path.join(args.output_dir, "lightning_logs", "version_0", "hparams.yaml")
        model_ckpt = best_ckpt_on_dev
        print(f">>> load model checkpoint: {model_ckpt}")
        trained_model = FinetunePLMTask.load_from_checkpoint(keep_label_lst=full_id_label_lst, checkpoint_path=model_ckpt, hparams_file=hparams_file,
                                                             map_location=None, batch_size=1, max_length=args.max_length, workers=0, )
        dataloader = get_dataloader(args, tokenizer, args.data_prefix, full_id_label_lst,
                                    pretrian_model=args.pretrained_plm_model, dist_sign=dist_sign)

        pred_logits_results = []
        trained_model.model.to(working_device).eval()
        for batch in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.no_grad():
                input_ids, token_type_ids, attention_mask = input_ids.to(working_device), token_type_ids.to(working_device), attention_mask.to(working_device)
                output_logits = trained_model.model(input_ids, token_type_ids, attention_mask)
                output_logits = output_logits.detach().cpu().numpy()
                pred_logits_results.append(output_logits)
        pred_logits_tensor = np.squeeze(np.stack(pred_logits_results, axis=0))
        os.makedirs(save_eval_dir, exist_ok=True)
        os.system(f"chmod -R 777 {save_eval_dir}")
        path_to_save_np_results = os.path.join(save_eval_dir, f"{dist_sign}_{idx}_logits.npy")
        np.save(path_to_save_np_results, pred_logits_tensor)
        print(f">>> save pred probabilities to {path_to_save_np_results}")


def collect_single_model_confidence_score(save_eval_dir, confidence_strategy="msp", temperature_value=1.0):
    load_id_np_file = os.path.join(save_eval_dir, f"id_logits.npy")
    id_logits_value = np.load(load_id_np_file)

    if confidence_strategy == "msp":
        id_avg_prob_matrix = np.exp(id_logits_value) / np.sum(np.exp(id_logits_value), axis=-1)
        print(f"check the shape of prob matrix {id_avg_prob_matrix.shape}")
        id_pred_label_idx_array = np.argmax(id_avg_prob_matrix, axis=-1)
        id_confidence_array = np.amax(id_avg_prob_matrix, axis=-1)
    elif confidence_strategy == "temp_scaling":
        id_confidence_array, id_pred_label_idx_array, id_prob_array = get_confidence_via_temperature_scale(id_logits_value, temperature_value)

    load_ood_np_file = os.path.join(save_eval_dir, f"ood_logits.npy")
    ood_logits_value = np.load(load_ood_np_file)
    if confidence_strategy == "msp":
        ood_avg_prob_matrix = np.exp(ood_logits_value) / np.sum(np.exp(ood_logits_value), axis=-1)
        print(f"check the shape of prob matrix {ood_avg_prob_matrix.shape}")
        ood_confidence_array = np.amax(ood_avg_prob_matrix, axis=-1)
    elif confidence_strategy == "temp_scaling":
        ood_confidence_array, ood_pred_label_idx_array, ood_prob_array = get_confidence_via_temperature_scale(ood_logits_value, temperature_value)
    return id_pred_label_idx_array, id_confidence_array, ood_confidence_array


def collect_kfolden_confidence_score(num_of_left_label, save_eval_dir, full_id_label_lst, confidence_strategy="msp", temperature_value=1.0):
    # collect id confidence scores
    id_pred_prob_lst = []
    for idx in range(0, len(full_id_label_lst), num_of_left_label):
        load_np_file = os.path.join(save_eval_dir, f"id_{idx}.npy")
        np_value = np.load(load_np_file)
        id_pred_prob_lst.append(np_value)
    id_sum_prob_matrix = np.sum(id_pred_prob_lst, axis=0)
    id_avg_prob_matrix = id_sum_prob_matrix / float(len(full_id_label_lst))
    print(f">>> check the value of sum_prob_matrix")
    print(f"{id_sum_prob_matrix[:2, :]}")
    id_pred_label_idx_array = np.argmax(id_avg_prob_matrix, axis=-1)

    if confidence_strategy == "msp":
        id_confidence_array = np.amax(id_avg_prob_matrix, axis=-1)

    # collect ood confidence scores
    ood_pred_prob_lst = []
    for idx in range(0, len(full_id_label_lst), num_of_left_label):
        load_np_file = os.path.join(save_eval_dir, f"ood_{idx}.npy")
        np_value = np.load(load_np_file)
        ood_pred_prob_lst.append(np_value)
    ood_sum_prob_matrix = np.sum(ood_pred_prob_lst, axis=0)
    ood_avg_prob_matrix = ood_sum_prob_matrix / float(len(full_id_label_lst))

    if confidence_strategy == "msp":
        ood_confidence_array = np.amax(ood_avg_prob_matrix, axis=-1)

    return id_pred_label_idx_array, id_confidence_array, ood_confidence_array


def collect_ensemble_models_confidence_score(save_eval_dir, num_of_ensemble, confidence_strategy="msp", temperature_value=1.0):
    id_pred_logits_lst = []
    for idx in range(num_of_ensemble):
        load_np_file = os.path.join(save_eval_dir, f"id_{idx}_logits.npy")
        np_value = np.load(load_np_file)
        id_pred_logits_lst.append(np_value)

    if confidence_strategy == "msp":
        id_pred_prob_array = np.array([compute_softmax_np(logits_item) for logits_item in id_pred_logits_lst])
        id_sum_prob_matrix = np.sum(id_pred_prob_array, axis=0)
        id_avg_prob_matrix = id_sum_prob_matrix / float(num_of_ensemble)
        id_pred_label_idx_array = np.argmax(id_avg_prob_matrix, axis=-1)
        id_confidence_array = np.amax(id_avg_prob_matrix, axis=-1)

    # collect ood confidence scores
    ood_pred_logits_lst = []
    for idx in range(num_of_ensemble):
        load_np_file = os.path.join(save_eval_dir, f"ood_{idx}_logits.npy")
        np_value = np.load(load_np_file)
        ood_pred_logits_lst.append(np_value)

    if confidence_strategy == "msp":
        ood_pred_prob_array = np.array([compute_softmax_np(logits_item) for logits_item in ood_pred_logits_lst])
        ood_sum_prob_matrix = np.sum(ood_pred_prob_array, axis=0)
        ood_avg_prob_matrix = ood_sum_prob_matrix / float(num_of_ensemble)
        ood_confidence_array = np.amax(ood_avg_prob_matrix, axis=-1)

    return id_pred_label_idx_array, id_confidence_array, ood_confidence_array

def evaluate_performances(id_pred_label_idx_array, id_dataset_label_idx_lst, id_confidence_array, ood_confidence_array, id_label_lst):
    num_id_labels = len(id_label_lst)
    id_acc = compute_accuracy(id_pred_label_idx_array, id_dataset_label_idx_lst)
    fpr_value = compute_fpr_tpr95(id_confidence_array, ood_confidence_array, num_id_labels)
    auroc_value = compute_auroc(id_confidence_array, ood_confidence_array, num_id_labels)
    id_aupr_value = compute_id_aupr(id_confidence_array, ood_confidence_array, num_id_labels)
    ood_aupr_value = compute_ood_aupr(id_confidence_array, ood_confidence_array, num_id_labels)
    mini_ood_error = compute_minimum_ood_detection_error_rate(id_confidence_array, ood_confidence_array, num_id_labels)

    return id_acc, fpr_value, auroc_value, id_aupr_value, ood_aupr_value, mini_ood_error


def collect_single_model_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, dist_sign="id"):
    file_saving_best_ckpt_path = os.path.join(args.output_dir, "best_ckpt_on_dev.txt")
    with open(file_saving_best_ckpt_path, "r") as f:
        best_ckpt_on_dev = f.read().strip()

    hparams_file = os.path.join(args.output_dir, "lightning_logs", "version_0", "hparams.yaml")
    model_ckpt = best_ckpt_on_dev
    print(f">>> load model checkpoint: {model_ckpt}")
    trained_model = FinetunePLMTask.load_from_checkpoint(keep_label_lst=full_id_label_lst, checkpoint_path=model_ckpt,
                                                         hparams_file=hparams_file, map_location=None, batch_size=1,
                                                         max_length=args.max_length, workers=0, )
    dataloader = get_dataloader(args, tokenizer, args.data_prefix, full_id_label_lst,
                                pretrian_model=args.pretrained_plm_model, dist_sign=dist_sign)

    pred_logits_results = []
    trained_model.model.to(working_device).eval()
    for batch in tqdm(dataloader):
        input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], \
                                                                                batch[4]
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask = input_ids.to(working_device), token_type_ids.to(
                working_device), attention_mask.to(working_device)
            output_logits = trained_model.model(input_ids, token_type_ids, attention_mask)
            output_logits = output_logits.detach().cpu().numpy()
            pred_logits_results.append(output_logits)

    pred_logits_tensor = np.squeeze(np.stack(pred_logits_results, axis=0))
    num_data = pred_logits_tensor.shape[0]
    print(f">>> check the shape of pred_prob {pred_logits_tensor.shape}")
    os.makedirs(save_eval_dir, exist_ok=True)
    os.system(f"chmod -R 777 {save_eval_dir}")
    path_to_save_np_results = os.path.join(save_eval_dir, f"{dist_sign}_logits.npy")
    np.save(path_to_save_np_results, pred_logits_tensor)
    print(f">>> save pred probabilities to {path_to_save_np_results}")


def main():
    # parse input arguments
    parser = get_parser()
    args = parser.parse_args()

    # use cuda or not
    use_cuda = torch.cuda.is_available()
    working_device = torch.device(f"cuda:{torch.cuda.current_device()}" if use_cuda else "cpu")
    print(f">>> use cuda: {use_cuda}; working device: {working_device}")

    # load tokenizer
    full_id_label_lst = get_labels(data_sign=args.data_name, dist_sign="id")
    num_id_labels = len(full_id_label_lst)
    print(f">>> TRAIN labels : {full_id_label_lst}")
    if args.pretrained_plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_config_dir, use_fast=False, do_lower_case=args.do_lower_case)
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

            print("%%%"*60)
            print(f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                  f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
            print("%%%"*60)
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

                print("%%%" * 60)
                print(
                    f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                    f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
                print("%%%" * 60)

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

                print("%%%" * 60)
                print(
                    f"Confidence Strategy is: > {strategy_item} \n > ID-ACC : {id_acc} \n > FPR@TPR95 : {fpr_value} \n"
                    f"> AUROC : {auroc_value} \n > ID-AUPR : {id_aupr_value} \n > OOD-AUPR : {ood_aupr_value} \n > MINI-OOD-ERROR : {mini_ood_error}")
                print("%%%" * 60)
        else:
            raise ValueError("Please append --enable_leave_label_out in you command . ")


if __name__ == "__main__":
    main()