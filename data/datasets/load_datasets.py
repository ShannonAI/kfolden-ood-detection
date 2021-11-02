#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: load_datasets.py

from torch.utils.data import DataLoader, SequentialSampler

from data.datasets.nn_doc_dataset import NNDocDataset
from data.datasets.plm_doc_dataset import PLMDocDataset
from data.datasets.collate_functions import collate_plm_to_max_length, collate_nn_to_max_length


def get_dataloader(input_arguments, tokenizer, mode, keep_label_lst, pretrian_model=True, dist_sign="id"):
    batch_size = input_arguments.eval_batch_size
    if pretrian_model:
        dataset = PLMDocDataset(input_arguments, tokenizer, distribution_type=dist_sign, mode=mode, keep_label_lst=keep_label_lst)
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_plm_to_max_length)
    else:
        dataset = NNDocDataset(input_arguments, max_seq_length=input_arguments.max_length, keep_label_lst=keep_label_lst,
                               vocab_file=input_arguments.vocab_file, distribution_type=dist_sign, do_lower_case=input_arguments.do_lower_case)
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=batch_size, collate_fn=collate_nn_to_max_length)

    return dataloader