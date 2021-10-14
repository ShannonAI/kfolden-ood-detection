#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: plm_doc_processor.py 

import os
import csv
from collections import namedtuple
from data.datasets.text_fields import non_semantic_shift_fields, semantic_shift_fields


class PLMDocProcessor:
    def __init__(self, data_dir, is_semantic_shift=True, dataset_name="agnews_ext"):
        self.data_dir = data_dir
        if is_semantic_shift:
            self.data_object = semantic_shift_fields[dataset_name]
            self.text_fields = semantic_shift_fields[dataset_name]._fields
        else:
            self.data_object = non_semantic_shift_fields[dataset_name]
            self.text_fields = non_semantic_shift_fields[dataset_name]._fields

    def get_train_examples(self, dist_sign="id"):
        train_file = os.path.join(self.data_dir, "train", "train.csv")
        train_data_examples = self._read_csv(train_file)
        return self._create_examples(train_data_examples, f"{dist_sign}-train")

    def get_dev_examples(self, dist_sign="id"):
        dev_file = os.path.join(self.data_dir, "dev", f"{dist_sign}_dev.csv")
        dev_data_examples = self._read_csv(dev_file)
        return self._create_examples(dev_data_examples, f"{dist_sign}-dev")

    def get_test_examples(self, dist_sign="id"):
        test_file = os.path.join(self.data_dir, "test", f"{dist_sign}_test.csv")
        test_data_examples = self._read_csv(test_file)
        return self._create_examples(test_data_examples, f"{dist_sign}-test")

    def _create_examples(self, data_lines, data_sign):
        examples = []
        for idx, data_obj in enumerate(data_lines):
            for text_field in self.text_fields:
                examples.append(data_obj)
        return examples

    def _read_csv(self, input_file):
        with open(input_file, "r") as csvfile:
            csv_reader = csv.DictReader(csvfile)
            return list(csv_reader)





