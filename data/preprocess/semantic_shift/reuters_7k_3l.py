#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: reuters_7k_3l.py

import os
import csv
import random
import argparse
import collections
from utils.random_seed import set_random_seed
set_random_seed(2333)
from data.data_utils.clean_text import tokenize_and_clean_text_str


def get_id_and_ood_data_statistic():
    data_label = {
        "ood_test_per_label": 1000,
        "ood_test": 4000,
        "id_test_per_label": 1000,
        "id_test": 4000,
        "ood_dev_per_label": 1000,
        "ood_dev": 4000,
        "id_dev_per_label": 1000,
        "id_dev": 4000,
        "train_per_label": 29000,
        "train":  116000
    }
    return data_label

def split_id_ood_distribution_strategy(agnews_train_obj_dict, agnews_dev_obj_dict, agnews_test_obj_dict, agcorpus_obj_dict):
    data_statistic_dict = get_id_and_ood_data_statistic()


def get_argument_parser():
    parser = argparse.ArgumentParser(description="return agnews-ext argument parser.")
    parser.add_argument("--ag_news_data_dir", type=str, required=True, help="yahoo_answers data dir")
    parser.add_argument("--ag_corpus_data_file", type=str, required=True, help="ag_corpus data dir")
    parser.add_argument("--save_data_dir", type=str, required=True, help="")
    parser.add_argument("--save_original_data_to_csv", action="store_true")

    return parser


def save_id_ood_data(save_data_dir, data_file_name, data_object_dict):
    # train_dir, dev_id_dir, test_id_dir, dev_ood_dir, test_ood_dir
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    data_file_path = os.path.join(save_data_dir, data_file_name)
    data_counter = 0
    with open(data_file_path, mode="w", encoding="utf-8", newline="\n") as w_csv_file:
        fieldnames = ['label', 'data']
        writer = csv.DictWriter(w_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data_obj_key in data_object_dict.keys():
            for data_item in data_object_dict[data_obj_key]:
                data_counter += 1
                cleaned_title = tokenize_and_clean_text_str(data_item.title)
                cleaned_desc = tokenize_and_clean_text_str(data_item.description)
                text_content = cleaned_title + " " + cleaned_desc
                writer.writerow({'label': data_item.topic,
                                 'data': repr(text_content)})

    print(f">>> save file to : {data_file_path}")
    print(f">>> the number of record is : {data_counter}")

def main():
    parser = get_argument_parser()
    input_arguments = parser.parse_args()

    agnews_train_dict, agnews_dev_dict, agnews_test_dict = load_agnews_data(input_arguments.ag_news_data_dir)
    agcorpus_data_dict = load_agcorpus_data(input_arguments.ag_corpus_data_file)
    train_id_dict, dev_id_dict, test_id_dict, dev_ood_dict, test_ood_dict = split_id_ood_distribution_strategy(
        agnews_train_dict, agnews_dev_dict, agnews_test_dict, agcorpus_data_dict)

    save_train_dir = os.path.join(input_arguments.save_data_dir, "train")
    save_dev_dir = os.path.join(input_arguments.save_data_dir, "dev")
    save_test_dir = os.path.join(input_arguments.save_data_dir, "test")

    print(f"$$$ TRAIN: ")
    save_id_ood_data(save_train_dir, "train.csv", train_id_dict)
    print(f"$$$ ID DEV: ")
    save_id_ood_data(save_dev_dir, "id_dev.csv", dev_id_dict)
    print(f"$$$ OOD DEV: ")
    save_id_ood_data(save_dev_dir, "ood_dev.csv", dev_ood_dict)
    print(f"$$$ ID TEST: ")
    save_id_ood_data(save_test_dir, "id_test.csv", test_id_dict)
    print(f"$$$ OOD TEST: ")
    save_id_ood_data(save_test_dir, "ood_test.csv", test_ood_dict)


if __name__ == "__main__":
    main()