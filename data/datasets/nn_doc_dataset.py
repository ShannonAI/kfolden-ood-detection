#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: datasets/nn_doc_dataset.py

import re
import os
import torch
from glob import glob
from torch.utils.data import Dataset
from data.data_utils.clean_text import remove_stop_and_lowfreq_words_func, clean_20newsgroup_data, tokenize_and_clean_text_str

def load_datafiles(data_repo, remove=None, label_lst=None, allow_ood=True, do_lower_case=True):
    data_content_lst = []
    data_label_lst = []

    data_files_lst = glob(os.path.join(data_repo, "*/*"))
    for data_file in data_files_lst:
        data_label = data_file.split("/")[-2]
        if not allow_ood and (data_label not in label_lst):
            continue

        data_label_lst.append(data_label)
        with open(data_file, "r", encoding="utf-8") as f:
            data_item = f.read()
            data_item = clean_20newsgroup_data(data_item, remove=remove)
        if do_lower_case:
            data_item = data_item.lower()
        data_content_lst.append(data_item)
    return data_content_lst, data_label_lst

class NNDocDataset(Dataset):
    def __init__(self, data_dir: str, data_prefix: str, max_length=512, tokenizer=None, allow_ood=True, vocab_file="./vocab.txt",
                 remove=["headers", "footers", "quotes"], label_files="./benchmark_labels.txt", do_lower_case=False,
                 stopwords_file="/home/lixiaoya/nltk_data/corpora/stopwords/english", remove_stop_and_lowfreq_words=True):
        """
        Args:
            tokenizer: should be an instance of AutoTokenizer or None
        """
        super(NNDocDataset, self).__init__()

        with open(label_files, "r") as f:
            self.label = [label.strip() for label in f.readlines()]
            self.label2idx = {lab: idx for idx, lab in enumerate(self.label)}

        self.data_repo = os.path.join(data_dir, data_prefix)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        if self.tokenizer is None:
            self.vocab_token2idx = {}
            with open(vocab_file, "r") as f:
                vocab_lines = f.readlines()
            for vocab_idx, vocab_item in enumerate(vocab_lines):
                vocab_item = vocab_item.strip()
                self.vocab_token2idx[vocab_item] = vocab_idx

        else:
            self.vocab_token2idx = None

        self.data_content_lst, self.data_label_lst = load_datafiles(self.data_repo, remove=remove, label_lst=self.label, allow_ood=allow_ood, do_lower_case=do_lower_case)
        if remove_stop_and_lowfreq_words:
            self.data_content_lst, self.word2freq_dict = remove_stop_and_lowfreq_words_func(self.data_content_lst, stopwords_file=stopwords_file)

    def __len__(self):
        return len(self.data_content_lst)

    def __getitem__(self, item):
        data_content_str = self.data_content_lst[item]
        label_idx_tensor = torch.tensor(self.label2idx[self.data_label_lst[item]], dtype=torch.long)
        data_tokens_idx_lst = []
        # clean data string, following https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        data_content_str = tokenize_and_clean_text_str(data_content_str)
        data_content_tokens = data_content_str.split(" ")

        for data_token in data_content_tokens:
            if data_token not in self.vocab_token2idx.keys():
                data_tokens_idx_lst.append(self.vocab_token2idx["<UNK>"])
            else:
                data_tokens_idx_lst.append(self.vocab_token2idx[data_token])

        if len(data_tokens_idx_lst) > self.max_length:
            data_tokens_idx_lst = data_tokens_idx_lst[: self.max_length]

        tokens_ids_tensor = torch.tensor(data_tokens_idx_lst, dtype=torch.long)

        return {"token_input_ids": tokens_ids_tensor, "label": label_idx_tensor}

