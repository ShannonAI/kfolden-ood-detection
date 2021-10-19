#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: plm_doc_processor.py 

import os
import csv
import torch
from collections import namedtuple
from torch.utils.data import TensorDataset

DocDataFeature = namedtuple("DocDataFeature", ["input_ids", "attention_mask", "token_type_ids", "label"])


class PLMDocProcessor:
    def __init__(self, data_dir, dataset_name="agnews_ext"):
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


def convert_examples_to_features(data_sign, data_example_lst, tokenizer, max_length, keep_label_lst):

    label_map = {label: i for i, label in enumerate(keep_label_lst)}
    labels = [label_map[example["label"]] if example["label"] in keep_label_lst else -1 for example in data_example_lst]

    if data_sign == "agnews_ext":
        # data_type,topic,title,description
        batch_encoding = tokenizer([
            f"{example['title']} {example['description']}" for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    elif data_sign == "20news_6s":
        # merge_label,data,label
        batch_encoding = tokenizer([
            example["data"] for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    elif data_sign == "yahoo_agnews_five":
        # label,title,description
        batch_encoding = tokenizer([
            f"{example['title']} {example['description']}" for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    elif data_sign == "agnews_fl":
        # label,title,description
        batch_encoding = tokenizer([
            f"{example['title']} {example['description']}" for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    elif data_sign == "agnews_fm":
        # label,title,description
        batch_encoding = tokenizer([
            f"{example['title']} {example['description']}" for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    elif data_sign == "yahoo_answers_fm":
        # title,content,best_answer,label
        batch_encoding = tokenizer([
            f"{example['title']} {example['content']}" for example in data_example_lst],
            max_length=max_length, padding="max_length", truncation=True)
    else:
        raise ValueError("ERROR : NOT Existing data signature... ...")

    features = []
    for i in range(len(data_example_lst)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = DocDataFeature(**inputs, label=labels[i])
        features.append(feature)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return features, dataset





