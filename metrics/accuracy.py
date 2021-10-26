#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: accuracy.py

import numpy as np


def compute_accuracy(pred_labels, gold_labels):
    pred_labels_np = np.array(pred_labels)
    gold_labels_np = np.array(gold_labels)

    num_of_data = pred_labels_np.shape[0]
    accuracy_value = (pred_labels_np == gold_labels_np).sum() / float(num_of_data)

    return accuracy_value


