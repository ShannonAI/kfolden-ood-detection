#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: model_calibration.py
# the code is adopt from: https://github.com/facebookresearch/odin/blob/main/code/calMetric.py

import numpy as np


def compute_fpr_tpr95(id_confidence_array: np.array, ood_confidence_array: np.array, num_id_labels: int):
    # return FPR value when true_positive rate reaches 0.95
    start, end = 1 / float(num_id_labels), 1
    gap = (end - start) / float(100000)
    total = 0.0
    fpr = 0.0
    num_id_data = id_confidence_array.shape[0]
    num_ood_data = ood_confidence_array.shape[0]
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(id_confidence_array >= delta)) / np.float(num_id_data)
        error2 = np.sum(np.sum(ood_confidence_array > delta)) / np.float(num_ood_data)
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fpr95_score_value = fpr / total
    return fpr95_score_value


def compute_auroc(id_confidence_array: np.array, ood_confidence_array: np.array, num_id_labels: int):
    start, end = 1 / float(num_id_labels), 1
    gap = (end - start) / float(100000)
    auroc_score = 0.0
    fpr_temp = 1.0
    num_id_data = id_confidence_array.shape[0]
    num_ood_data = ood_confidence_array.shape[0]

    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(id_confidence_array >= delta)) / np.float(num_id_data)
        fpr = np.sum(np.sum(ood_confidence_array > delta)) / np.float(num_ood_data)
        auroc_score += (-fpr + fpr_temp) * tpr
        fpr_temp = fpr
    auroc_score += fpr * tpr
    return auroc_score


def compute_id_aupr(id_confidence_array: np.array, ood_confidence_array: np.array, num_id_labels: int):
    start, end = 1 / float(num_id_labels), 1
    gap = (end - start) / float(100000)

    precisionVec = []
    recallVec = []
    id_aupr_score = 0.0
    recall_temp = 1.0

    num_id_data = id_confidence_array.shape[0]
    num_ood_data = ood_confidence_array.shape[0]

    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(id_confidence_array >= delta)) / np.float(num_id_data)
        fp = np.sum(np.sum(ood_confidence_array >= delta)) / np.float(num_ood_data)
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        id_aupr_score += (recall_temp - recall) * precision
        recall_temp = recall
    id_aupr_score += recall * precision
    return id_aupr_score


def compute_ood_aupr(id_confidence_array: np.array, ood_confidence_array: np.array, num_id_labels: int):
    start, end = 1 / float(num_id_labels), 1
    gap = (end - start) / float(100000)

    ood_aupr_score = 0.0
    recall_temp = 1.0
    num_id_data = id_confidence_array.shape[0]
    num_ood_data = ood_confidence_array.shape[0]
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(id_confidence_array < delta)) / np.float(num_id_data)
        tp = np.sum(np.sum(ood_confidence_array < delta)) / np.float(num_ood_data)
        if tp + fp == 0:
            break
        precision = tp / (tp + fp)
        recall = tp
        ood_aupr_score += (recall_temp - recall) * precision
        recall_temp = recall
    ood_aupr_score += recall * precision
    return ood_aupr_score


def compute_minimum_ood_detection_error_rate(id_confidence_array: np.array, ood_confidence_array: np.array, num_id_labels: int):
    start, end = 1 / float(num_id_labels), 1
    gap = (end - start) / float(100000)
    num_id_data = id_confidence_array.shape[0]
    num_ood_data = ood_confidence_array.shape[0]

    error_rate = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(id_confidence_array < delta)) / np.float(num_id_data)
        tmp_error = np.sum(np.sum(ood_confidence_array > delta)) / np.float(num_ood_data)
        error_rate = np.minimum(error_rate, (tpr + tmp_error) / 2.0)
    return error_rate