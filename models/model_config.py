#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file: model_config.py


from transformers import BertConfig


class BertForSequenceClassificationConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForSequenceClassificationConfig, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.num_labels = kwargs.get("num_labels", 2)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)

class ModelConfig:
    def __init__(self, **kwargs):
        # model config
        self.dropout = kwargs.get("dropout", 0.2)
        self.classifier_type = kwargs.get("classifier_type", "mlp")
        self.activate_func = kwargs.get("activate_func", "gelu")
        # emb config
        self.num_labels = kwargs.get("num_labels", 10)
        self.padding_idx = kwargs.get("padding_idx", 0)
        self.init_word_embedding = kwargs.get("init_word_embedding", "")
        self.freeze_word_embedding = kwargs.get("freeze_word_embedding", False)

class CNNTextClassificationConfig(ModelConfig):
    def __init__(self, **kwargs):
        super(CNNTextClassificationConfig, self).__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 128)
        self.embedding_size = kwargs.get("embedding_size", 128)
        self.hidden_size = kwargs.get("hidden_size", 128)
        self.num_layers = kwargs.get("num_layers", 1)
        self.num_kernels = kwargs.get("num_kernels", 3)
        self.kernel_size = kwargs.get("kernel_size", "3;4;5")
        self.conv_stride = kwargs.get("conv_stride", "1;1;1")
        self.pooling_strategy = kwargs.get("pooling_strategy", "max_pool")

class RNNTextClassificationConfig(ModelConfig):
    def __init__(self, **kwargs):
        super(RNNTextClassificationConfig, self).__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 128)
        self.embedding_size = kwargs.get("embedding_size", 128)
        self.hidden_size = kwargs.get("hidden_size", 128)
        self.num_layers = kwargs.get("num_layers", 2)
        self.bidirectional = kwargs.get("bidirectional", False)
        self.batch_first = kwargs.get("batch_first", True)
        self.rnn_dropout = kwargs.get("rnn_dropout", 0.1)
        self.rnn_activate_func = kwargs.get("rnn_activate_func", "relu")
        self.pooling_strategy = kwargs.get("pooling_strategy", "avg_pool")
        self.rnn_cell_type = kwargs.get("rnn_cell_type", "rnn")

