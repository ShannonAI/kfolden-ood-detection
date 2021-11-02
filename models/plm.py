#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: plm.py
# notice: based on https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/modeling_roberta.py
#         and https://github.com/huggingface/transformers/blob/v3.4.0/src/transformers/modeling_bert.py
# may need to change when using transformers!=3.4.0

import torch.nn as nn
from torch import Tensor
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from models.classifier import truncated_normal_
from models.model_config import BertForSequenceClassificationConfig, RobertaForSequenceClassificationConfig


class BertForSequenceClassification(BertPreTrainedModel):
    """Fine-tune BERT model for text classification."""
    def __init__(self, config: BertForSequenceClassificationConfig,):
        super(BertForSequenceClassification, self).__init__(config)
        self.bert_config = config
        self.bert = BertModel(config,)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_classifier.weight = truncated_normal_(self.cls_classifier.weight, mean=0, std=0.02)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, output_hidden_states=None, return_dict=None):
        """
        Args:
            inputs_ids: input tokens, tensor of shape [batch_size, seq_len].
            token_type_ids: 1 for text_b tokens and 0 for text_a tokens. tensor of shape [batch_size, seq_len].
            attention_mask: 1 for non-[PAD] tokens and 0 for [PAD] tokens. tensor of shape [batch_size, seq_len].
            output_hidden_states: (default None)
            return_dict: (default None)
        Returns:
            cls_outputs: output logits for the [CLS] token. tensor of shape [batch_size, num_labels].
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 output_hidden_states=output_hidden_states, return_dict=return_dict)
        if not return_dict:
            bert_cls_output = bert_outputs[1]
        else:
            bert_cls_output = bert_outputs.pooler_output
        # bert_cls_output is in the shape of (batch_size, number_of_classes)
        bert_cls_output = self.dropout(bert_cls_output)
        cls_logits = self.cls_classifier(bert_cls_output)

        if not output_hidden_states:
            return cls_logits
        else:
            all_layer_hidden_states = bert_outputs.hidden_states
            # bert_embedding + bert_encoder_layers
            # <number of layers> and every tensor is in the shape of (batch_size, input_seq_len, model_hidden_size)
            return cls_logits, all_layer_hidden_states


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    """Fine-tune RoBERTa model for text classification."""
    def __init__(self, config: RobertaForSequenceClassificationConfig,):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.roberta_config = config
        self.roberta = RobertaModel(config,)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_classifier.weight = truncated_normal_(self.cls_classifier.weight, mean=0, std=0.02)
        self.init_weights()

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor, output_hidden_states=None, return_dict=None):
        """
        Args:
            inputs_ids: input tokens, tensor of shape [batch_size, seq_len].
            token_type_ids: 1 for text_b tokens and 0 for text_a tokens. tensor of shape [batch_size, seq_len].
            attention_mask: 1 for non-[PAD] tokens and 0 for [PAD] tokens. tensor of shape [batch_size, seq_len].
        Returns:
            cls_outputs: output logits for the [CLS] token. tensor of shape [batch_size, num_labels].
        """
        roberta_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                       output_hidden_states=output_hidden_states, return_dict=return_dict,)

        if not return_dict:
            roberta_cls_output = roberta_outputs[1]
        else:
            roberta_cls_output = roberta_outputs.pooler_output
        # roberta_cls_output is in the shape of (batch_size, number_of_classes)

        roberta_cls_output = self.dropout(roberta_cls_output)
        cls_logits = self.cls_classifier(roberta_cls_output)

        if not output_hidden_states:
            return cls_logits
        else:
            all_layer_hidden_states = roberta_outputs.hidden_states
            # roberta_embedding + roberta_encoder_layers
            # <number of layers> and every tensor is in the shape of (batch_size, input_seq_len, model_hidden_size)
            return cls_logits, all_layer_hidden_states
