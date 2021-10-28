#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: data_statistic.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from data.datasets.nn_doc_dataset import NNDocDataset
from data.datasets.plm_doc_dataset import PLMDocDataset
from data.datasets.label_fields import get_labels
from data.datasets.plm_doc_processor import PLMDocProcessor
from data.datasets.nn_doc_processor import NNDocProcessor
from transformers import AutoTokenizer


class DataArgument:
    def __init__(self):
        self.data_name = "agnews_fl"
        self.data_dir = "/data/lixiaoya/datasets/kfolden/agnews_fl"
        self.max_length = 1000


def do_data_statistic(model_dir):
    input_args = DataArgument()
    label_lst = get_labels(input_args.data_name, dist_sign="id")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, do_lower_case=False)
    data_processor = PLMDocProcessor(input_args.data_dir, dataset_name=input_args.data_name)
    for data_mode in ["train", "dev", "test"]:
        input_ids_len_lst = []
        # args, tokenizer, distribution_type="id", mode="train", keep_label_lst=[]
        data_instances = data_processor.get_test_examples()
        for data_feature in data_instances:
            data_len = data_feature.input_ids
            print(data_len)


if __name__ == "__main__":
    model_dir = "/data/lixiaoya/models/bert_uncased_base"
    do_data_statistic(model_dir)


