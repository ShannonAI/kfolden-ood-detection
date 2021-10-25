#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: evaluate_saved_ckpt.py

import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from utils.random_seed import set_random_seed
set_random_seed(2333)

from transformers import AutoTokenizer
from data.datasets.label_fields import get_labels
from data.datasets.nn_doc_dataset import NNDocDataset
from data.datasets.plm_doc_dataset import PLMDocDataset
from data.datasets.collate_functions import collate_plm_to_max_length, collate_nn_to_max_length
from task.train_nn import TrainNNTask
from task.finetune_plm import FinetunePLMTask
from metrics.metric_funcs import get_fpr_95, get_auroc

def get_parser():
    parser = argparse.ArgumentParser(description="inference the model output.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bert_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--is_chinese", action="store_true")
    parser.add_argument("--save_output_dir", type=str, default="")
    parser.add_argument("--data_name", type=str, choices=["ontonotes4", "msra", "conll03", "ace04", "ace05"],
                        default="conll03")
    parser.add_argument("--data_distribution", type=str, default="id")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--enable_leave_label_out", action="store_true", help="leave label out")
    parser.add_argument("--num_of_left_label", default=0, type=int, help="number of labels as ood data distribution")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--eval_mode", type=str, default="test")
    return parser

def get_dataloader(input_arguments, tokenizer, mode, keep_label_lst):
    dataset = PLMDocDataset(input_arguments, tokenizer, mode=mode, keep_label_lst=keep_label_lst)
    data_sampler = SequentialSampler(dataset)
    batch_size = input_arguments.eval_batch_size
    dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_plm_to_max_length)
    return dataloader

def main():
    parser = get_parser()
    args = parser.parse_args()

    full_id_label_lst = get_labels(data_sign=args.data_name, dist_sign="id")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=False, do_lower_case=args.do_lower_case)
    save_eval_dir = os.path.join(args.output_dir, "eval",)
    if args.enable_leave_label_out and args.data_distribution == "id":
        pred_prob_lst = []
        for idx in range(0, len(full_id_label_lst), args.num_of_left_label):
            left_label_lst = [item for item_idx, item in enumerate(full_id_label_lst) if item_idx in range(idx, idx+args.num_of_left_label)]
            keep_label_lst = [item for item_idx, item in enumerate(full_id_label_lst) if item_idx not in range(idx, idx+args.num_of_left_label)]
            sub_output_dir = os.path.join(args.output_dir, f"{idx}")

            file_saving_best_ckpt_path = os.path.join(sub_output_dir, "best_ckpt_on_dev.txt")
            with open(file_saving_best_ckpt_path, "r") as f:
                best_ckpt_on_dev = f.read().strip()

            hparams_file = os.path.join(sub_output_dir, "lightning_logs", "version_0", "hparams.yaml")
            model_ckpt = best_ckpt_on_dev
            trained_model = FinetunePLMTask.load_from_checkpoint(checkpoint_path=model_ckpt, hparams_file=hparams_file,
                                                                 map_location=None, batch_size=1,
                                                                 max_length=args.max_length, workers=0)
            dataloader = get_dataloader(args, tokenizer, args.eval_mode, keep_label_lst)

            pred_prob_results = []
            trained_model.model.eval()
            for batch in dataloader:
                input_ids, token_type_ids, attention_mask, id_label_mask, gold_labels = batch[0], batch[1], batch[2], batch[3], batch[4]
                with torch.no_grad():
                    output_logits = trained_model.model(input_ids, token_type_ids, attention_mask)
                    output_probs = F.softmax(output_logits, dim=-1).detach().cpu().numpy()
                    pred_prob_results.append(output_probs)
            pred_prob_tensor = np.stack(pred_prob_results, axis=0)
            num_data = pred_prob_tensor.shape[0]
            pad_pred_prob = np.zeros((num_data, len(full_id_label_lst)), dtype=float)
            pad_pred_prob[:, idx+1:] = pred_prob_tensor
            os.makedirs(save_eval_dir, exist_ok=True)
            os.system(f"chmod -R 777 {save_eval_dir}")
            path_to_save_np_results = os.path.join(save_eval_dir, f"{idx}.npy")
            np.save(path_to_save_np_results, pad_pred_prob)
            print(f">>> save pred probabilities to {path_to_save_np_results}")

        for idx in range(0, len(full_id_label_lst), args.num_of_left_label):
            load_np_file = os.path.join(path_to_save_np_results, f"{idx}.npy")
            np_value = np.load(load_np_file)
            pred_prob_lst.append(np_value)
        sum_prob_matrix = np.sum(pred_prob_lst, axis=0)
        avg_prob_matrix = sum_prob_matrix / float(len(full_id_label_lst))
        fpr95_value = get_fpr_95(avg_prob_matrix)
        auroc_value = get_auroc(avg_prob_matrix)

        print(f"AUROC: {auroc_value}; FPR-95: {fpr95_value}")


if __name__ == "__main__":
    main()