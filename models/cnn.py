#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: models/cnn.py

import torch
import numpy as np
from torch import nn
from models.classifier import MultiLayerPerceptronClassifier
from models.model_config import CNNTextClassificationConfig


class CNNForTextClassification(nn.Module):
    def __init__(self, config: CNNTextClassificationConfig):
        super(CNNForTextClassification, self).__init__()
        self.model_config = config
        # embedding
        if len(config.init_word_embedding) > 1:
            embedding_weight = torch.tensor(np.load(config.init_word_embedding), dtype=torch.float)
            # embedding_weight should be 2-d float tensor.
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=config.freeze_word_embedding,
                                                          padding_idx=config.padding_idx, )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.padding_idx)

        # cnn layer
        self.cnn_kernel_sizes = [int(kernel) for kernel in config.kernel_size.split(";")]
        self.conv_stride = [int(stride) for stride in config.conv_stride.split(";")]
        assert len(self.cnn_kernel_sizes) == config.num_kernels and len(self.conv_stride) == config.num_kernels
        self.cnn_layers = []
        for idx in range(config.num_kernels):
            kernel_size = self.cnn_kernel_sizes[idx]
            conv_stride = self.conv_stride[idx]
            cnn_layer = nn.Conv1d(config.embedding_size, config.hidden_size, kernel_size, stride=conv_stride,
                                  padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros').cuda()
            self.cnn_layers.append(cnn_layer)
        # pooling layer
        self.pooling_strategy = config.pooling_strategy
        # classifier
        self.classifier = MultiLayerPerceptronClassifier(hidden_size=config.num_kernels * config.hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_sequence):
        """
        Args:
            input_sequence: LongTensor, shape of (batch_size, seq_len)
        """
        input_seq_embeddings = self.embedding(input_sequence)
        input_seq_embeddings_permute = input_seq_embeddings.permute(0, 2, 1) # (batch_size, embed_size, seq_len)
        cnns_outputs = []
        for cnn_layer in self.cnn_layers:
            cnn_output = nn.Tanh()(cnn_layer(input_seq_embeddings_permute)) # (batch_size, hidden_size, dynamic_size)
            output_size = cnn_output.shape[-1]

            if self.pooling_strategy == "avg_pool":
                pool_layer = nn.AvgPool1d(output_size, stride=output_size)
            elif self.pooling_strategy == "max_pool":
                pool_layer = nn.MaxPool1d(output_size, stride=output_size)
            else:
                raise ValueError

            cnn_layer_output = torch.squeeze(pool_layer(cnn_output), 2) # (batch_size, hidden_size)
            cnns_outputs.append(cnn_layer_output)

        cnns_outputs = torch.stack(cnns_outputs, dim=2,) # (batch_size, hidden_size, num_kernels)
        batch_size = cnns_outputs.shape[0]
        cnns_outputs = cnns_outputs.view(batch_size, -1)
        cnns_outputs = self.dropout(cnns_outputs)
        classifier_output = self.classifier(cnns_outputs)

        return classifier_output