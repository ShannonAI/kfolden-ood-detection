#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: collate_functions.py


import torch
from typing import List


def collate_nn_to_max_length(batch: List[List[torch.Tensor]], padding_idx=0) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            input_ids, token_mask, id_label_mask, label
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x["input_ids"].shape[0] for x in batch)
    output = []

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["input_ids"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["input_ids"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["token_mask"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["token_mask"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["id_label_mask"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["id_label_mask"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    output.append(torch.stack([x["label"] for x in batch]))
    return output


def collate_plm_to_max_length(batch: List[List[torch.Tensor]], padding_idx=0) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            input_ids, token_type_ids, attention_mask, id_label_mask, label
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x["input_ids"].shape[0] for x in batch)
    output = []

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["input_ids"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["input_ids"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["token_type_ids"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["token_type_ids"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    pad_output = torch.full([batch_size, max_length], padding_idx, dtype=batch[0]["attention_mask"].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx]["attention_mask"]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)

    output.append(torch.stack([x["id_label_mask"] for x in batch]))
    output.append(torch.stack([x["label"] for x in batch]))
    return output