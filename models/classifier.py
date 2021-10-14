#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: modules.py
# description:
# modules for building models.


import torch.nn as nn


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class BertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_to_labels_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Tanh()
        if config.truncated_normal:
            self.dense_layer.weight = truncated_normal_(self.dense_layer.weight, mean=0, std=0.02)
            self.dense_to_labels_layer.weight = truncated_normal_(self.dense_to_labels_layer.weight, mean=0, std=0.02)

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output


class MultiLayerPerceptronClassifier(nn.Module):
    def __init__(self, hidden_size=None, num_labels=None, activate_func="gelu"):
        super().__init__()
        self.dense_layer = nn.Linear(hidden_size, hidden_size)
        self.dense_to_labels_layer = nn.Linear(hidden_size, num_labels)
        if activate_func == "tanh":
            self.activation = nn.Tanh()
        elif activate_func == "relu":
            self.activation = nn.ReLU()
        elif activate_func == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError

    def forward(self, sequence_hidden_states):
        sequence_output = self.dense_layer(sequence_hidden_states)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dense_to_labels_layer(sequence_output)
        return sequence_output





