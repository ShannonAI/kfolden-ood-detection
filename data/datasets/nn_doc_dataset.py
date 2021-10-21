#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: datasets/nn_doc_dataset.py

import torch
from torch.utils.data import Dataset
from data.datasets.nn_doc_processor import NNDocProcessor, convert_examples_to_features


class NNDocDataset(Dataset):
    def __init__(self, args, mode="train", max_seq_length=512, keep_label_lst=[], vocab_file="./vocab.txt",
                 distribution_type="id", do_lower_case=False,):
        super(NNDocDataset, self).__init__()

        self.args = args
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.data_name = args.data_name
        self.keep_label_lst = keep_label_lst

        self.processor = NNDocProcessor(self.args.data_dir, dataset_name=self.data_name, )
        if self.mode == "dev":
            self.examples = self.processor.get_dev_examples(dist_sign=distribution_type)
        elif self.mode == "test":
            self.examples = self.processor.get_test_examples(dist_sign=distribution_type)
        else:
            self.examples = self.processor.get_train_examples(dist_sign=distribution_type)

        self.features = convert_examples_to_features(self.data_name, self.examples, vocab_file, self.max_seq_length,
                                                     self.do_lower_case, self.keep_label_lst)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature = self.features[i]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        token_mask = torch.tensor(feature.token_mask, dtype=torch.long)
        label = torch.tensor(feature.label, dtype=torch.long)
        id_label_mask = torch.tensor([1 if feature.label in self.keep_label_lst else 0], dtype=torch.long)

        inputs = {
            "input_ids": input_ids,
            "token_mask": token_mask,
            "id_label_mask": id_label_mask,
            "label": label,
        }
        return inputs

