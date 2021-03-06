#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: datasets/plm_doc_dataset.py

import torch
from typing import Dict
from torch.utils.data.dataset import Dataset
from data.datasets.plm_doc_processor import PLMDocProcessor, convert_examples_to_features


class PLMDocDataset(Dataset):
    def __init__(self, args, tokenizer, distribution_type="id", mode="train", keep_label_lst=[]):
        self.args = args
        self.mode = mode
        self.distribution_type = distribution_type
        self.data_name = args.data_name
        self.processor = PLMDocProcessor(self.args.data_dir, dataset_name=self.data_name)
        self.max_seq_length = self.args.max_length
        self.keep_label_lst = keep_label_lst

        if self.mode == "dev":
            self.examples = self.processor.get_dev_examples(dist_sign=distribution_type)
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples(dist_sign=distribution_type)
        else:
            self.examples = self.processor.get_train_examples(dist_sign=distribution_type)

        self.features, self.dataset, self.keep_label_map = convert_examples_to_features(self.data_name, self.examples, tokenizer, self.max_seq_length, keep_label_lst)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # convert to Tensors and build dataset
        feature = self.features[i]

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        label = torch.tensor(feature.label, dtype=torch.long)
        id_label_mask = torch.tensor([1 if feature.label in self.keep_label_map.values() else 0], dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "id_label_mask": id_label_mask,
            "label": label,
        }
        return inputs
