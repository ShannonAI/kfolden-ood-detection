#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: datasets/nn_doc_processor.py

import os
import csv
import torch
from collections import namedtuple
from torch.utils.data import TensorDataset

from data.data_utils.clean_text import remove_stop_and_lowfreq_words_func, clean_20newsgroup_data, tokenize_and_clean_text_str

DocDataFeature = namedtuple("DocDataFeature", ["input_ids", "token_mask", "label"])


class NNDocProcessor:
    def __init__(self, data_dir, dataset_name="agnews_ext", ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def get_train_examples(self, dist_sign="id"):
        train_file = os.path.join(self.data_dir, "train", "train.csv")
        train_data_examples = self._read_csv(train_file)
        return train_data_examples

    def get_dev_examples(self, dist_sign="id"):
        dev_file = os.path.join(self.data_dir, "dev", f"{dist_sign}_dev.csv")
        dev_data_examples = self._read_csv(dev_file)
        return dev_data_examples

    def get_test_examples(self, dist_sign="id"):
        test_file = os.path.join(self.data_dir, "test", f"{dist_sign}_test.csv")
        test_data_examples = self._read_csv(test_file)
        return test_data_examples

    def _read_csv(self, input_file):
        with open(input_file, "r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            return list(csv_reader)


def convert_examples_to_features(data_sign, data_example_lst, vocab_file, max_seq_length, do_lowercase, keep_label_lst, ignore_index=-100):
    vocab_token2idx = {}
    with open(vocab_file, "r") as f:
        vocab_lines = f.readlines()
    for vocab_idx, vocab_item in enumerate(vocab_lines):
        vocab_item = vocab_item.strip()
        vocab_token2idx[vocab_item] = vocab_idx

    label_map = {label: i for i, label in enumerate(keep_label_lst)}
    labels = [label_map[example["label"]] if example["label"] in keep_label_lst else ignore_index for example in data_example_lst]

    features = []
    for data_idx, data_example in enumerate(data_example_lst):
        data_tokens_idx_lst = []
        data_content_str = data_example["data"]
        # clean data string, following https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        data_content_str = tokenize_and_clean_text_str(data_content_str)
        data_content_tokens = data_content_str.split(" ")

        for data_token in data_content_tokens:
            if data_token not in vocab_token2idx.keys():
                data_tokens_idx_lst.append(vocab_token2idx["<UNK>"])
            else:
                data_tokens_idx_lst.append(vocab_token2idx[data_token])
        if len(data_tokens_idx_lst) > max_seq_length:
            data_tokens_idx_lst = data_tokens_idx_lst[: max_seq_length]
        token_mask = [1] * len(data_tokens_idx_lst)

        feature = DocDataFeature(input_ids=data_tokens_idx_lst, token_mask=token_mask, label=labels[data_idx])
        features.append(feature)

    return features




