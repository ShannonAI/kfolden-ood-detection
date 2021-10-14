#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: plm_doc_dataset.py

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset
from typing import Dict, Optional, List, Union
from data.datasets.text_fields import non_semantic_shift_fields, semantic_shift_fields



class PLMDocDataset(Dataset):
    def __init__(self, args, tokenizer: BertTokenizer, distribution_type="id", mode="train"):

        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.distribution_type = distribution_type
        self.processor = PLMDocProcessor(self.args.data_dir)
        self.max_seq_length = self.args.max_seq_length

        if self.mode == "dev":
            self.examples = self.processor.get_dev_examples()
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples()
        else:
            self.examples = self.processor.get_train_examples()

        self.features, self.dataset = mrpc_convert_examples_to_features(
            examples=self.examples,
            tokenizer=tokenizer,
            max_length=self.max_seq_length,
            label_list=MRPCProcessor.get_labels(),
            is_training= mode == "train",)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        label = torch.tensor(feature.label, dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": label
        }

        return inputs
