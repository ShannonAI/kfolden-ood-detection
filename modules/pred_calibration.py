#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: pred_calibration.py

import torch
import torch.nn.functional as F
from torch import nn


def enable_dropout_for_trained_models(model: nn.Module, dropout_module: nn.Module, value:bool=True):
    for child_name, child in model.named_children():
        if isinstance(child, dropout_module):
            child.train(value)
        else:
            enable_dropout_for_trained_models(child, dropout_module, value=value)


def get_confidence_via_temperature_scale(pred_logits: torch.tensor, temperature_value: float = 1):
    soften_pred_logits = pred_logits / temperature_value
    soften_pred_probs = F.softmax(soften_pred_logits, dim=-1)
    confidence_tuples = torch.max(soften_pred_probs, -1, keepdim=False)
    confidence_value = confidence_tuples.values
    confidence_indices = confidence_tuples.indices # a.k.a. the same as the argmax operation .
    return confidence_value, confidence_indices, soften_pred_probs


def get_confidence_via_max_softmax_prob(pred_logits: torch.tensor):
    pred_probs = F.softmax(pred_logits, dim=-1)
    confidence_tuples = torch.max(pred_probs, -1, keepdim=False)
    confidence_value = confidence_tuples.values
    confidence_indices = confidence_tuples.indices  # a.k.a. the same as the argmax operation .
    return confidence_value, confidence_indices  


