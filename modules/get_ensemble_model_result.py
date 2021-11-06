#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: get_ensemble_model_result.py

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from task.train_nn import TrainNNTask
from task.finetune_plm import FinetunePLMTask
from data.datasets.load_datasets import get_dataloader
from modules.pred_calibration import get_confidence_via_max_softmax_prob, get_confidence_via_temperature_scale, compute_softmax_np


def collect_ensemble_models_confidence_score(save_eval_dir, num_of_ensemble, confidence_strategy="msp", temperature_value=1.0):
    id_pred_logits_lst = []
    for idx in range(num_of_ensemble):
        load_np_file = os.path.join(save_eval_dir, f"id_{idx}_logits.npy")
        np_value = np.load(load_np_file)
        id_pred_logits_lst.append(np_value)

    id_pred_prob_array = np.array([compute_softmax_np(logits_item) for logits_item in id_pred_logits_lst])
    id_sum_prob_matrix = np.sum(id_pred_prob_array, axis=0, keepdims=True)
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
        ood_sum_prob_matrix = np.sum(ood_pred_prob_array, axis=0, keepdims=True)
        ood_avg_prob_matrix = ood_sum_prob_matrix / float(num_of_ensemble)
        ood_confidence_array = np.amax(ood_avg_prob_matrix, axis=-1)

    return id_pred_label_idx_array, id_confidence_array, ood_confidence_array


def collect_ensemble_models_pred_logits(args, tokenizer, full_id_label_lst, save_eval_dir, working_device, num_ensemble, dist_sign="id"):
    for idx in range(num_ensemble):
        print("$" * 60)
        print(f">>> Loading {idx} Model ...")
        sub_output_dir = os.path.join(args.output_dir, f"{idx}")

        file_saving_best_ckpt_path = os.path.join(sub_output_dir, "best_ckpt_on_dev.txt")
        with open(file_saving_best_ckpt_path, "r") as f:
            best_ckpt_on_dev = f.read().strip()
        hparams_file = os.path.join(args.output_dir, "lightning_logs", f"version_{idx}", "hparams.yaml")
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



