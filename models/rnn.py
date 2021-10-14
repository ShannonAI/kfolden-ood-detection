#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: models/rnn.py


import torch
import numpy as np
from torch import nn
from models.model_config import RNNTextClassificationConfig
from models.classifier import MultiLayerPerceptronClassifier

class RNNForTextClassification(nn.Module):
    def __init__(self, config: RNNTextClassificationConfig):
        super(RNNForTextClassification, self).__init__()
        self.model_config = config
        # embedding
        if len(config.init_word_embedding) > 1:
            embedding_weight = torch.tensor(np.load(config.init_word_embedding), dtype=torch.float)
            # embedding_weight should be 2-d float tensor.
            self.embedding = nn.Embedding.from_pretrained(embedding_weight, freeze=config.freeze_word_embedding,
                                                          padding_idx=config.padding_idx,)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.padding_idx)

        if config.rnn_cell_type == "rnn":
            self.rnn_layers = nn.RNN(config.embedding_size, config.hidden_size, config.num_layers,
                                     nonlinearity=config.rnn_activate_func, batch_first=config.batch_first,
                                     dropout=config.rnn_dropout, bidirectional=config.bidirectional)
        elif config.rnn_cell_type == "lstm":
            self.rnn_layers = nn.LSTM(config.embedding_size, config.hidden_size, config.num_layers, bias=True,
                                      batch_first=config.batch_first, dropout=config.rnn_dropout, bidirectional=config.bidirectional)
        elif config.rnn_cell_type == "gru":
            self.rnn_layers = nn.GRU(config.embedding_size, config.hidden_size, config.num_layers, bias=True,
                                     batch_first=config.batch_first, dropout=config.rnn_dropout, bidirectional=config.bidirectional)
        else:
            raise ValueError("RNN_CELL_TYPE should be one of [rnn, lstm, gru]")
        hidden_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.classifier = MultiLayerPerceptronClassifier(hidden_size=hidden_size, num_labels=config.num_labels, activate_func=config.activate_func)
        self.dropout = nn.Dropout(config.dropout)
        self.pooling_strategy = config.pooling_strategy

    def forward(self, input_sequence):
        """
        Args:
            input_sequence: LongTensor, shape of (batch_size, seq_len)
        """
        input_seq_embeddings = self.embedding(input_sequence) # (batch_size, seq_len)
        rnn_seq_features, rnn_state_outputs = self.rnn_layers(input_seq_embeddings)
        # rnn_seq_features - > (batch_size, seq_len, num_directional * hidden_size)
        # rnn_state_output -> (num_directional * 2, batch_size, hidden_size)

        rnn_seq_features_permute = rnn_seq_features.permute(0, 2, 1)
        seq_len = rnn_seq_features_permute.shape[-1]

        if self.pooling_strategy == "avg_pool":
            pool_layer = nn.AvgPool1d(seq_len, stride=seq_len)
        elif self.pooling_strategy == "max_pool":
            pool_layer = nn.MaxPool1d(seq_len, stride=seq_len)
        else:
            raise ValueError

        seq_features = torch.squeeze(pool_layer(rnn_seq_features_permute), 2)  # (batch_size, hidden_size)
        seq_features = self.dropout(seq_features)
        classifier_output = self.classifier(seq_features)

        return classifier_output