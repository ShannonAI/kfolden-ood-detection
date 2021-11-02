#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: get_single_model_result.py

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from task.train_nn import TrainNNTask
from task.finetune_plm import FinetunePLMTask
from data.datasets.load_datasets import get_dataloader
from modules.pred_calibration import get_confidence_via_max_softmax_prob, get_confidence_via_temperature_scale, compute_softmax_np


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
