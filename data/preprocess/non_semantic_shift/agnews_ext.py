#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: agnews_ext.py

import os
import csv
import random
import argparse
import collections
from utils.random_seed import set_random_seed
set_random_seed(2333)
from data.data_utils.clean_text import tokenize_and_clean_text_str

AGCorpus = collections.namedtuple("AGCorpus",["name", "topic", "title", "description"])
AGNews = collections.namedtuple("AGNews",["name", "data_type", "topic", "title", "description"])

def load_agcorpus_data(data_file):
    # <category>Business</category>
    # <description>Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.</description>
    # <title>Wall St. Bears Claw Back Into the Black (Reuters)</title>

    with open(data_file, "r") as f:
        agcorpus_datalines = [data_item.strip() for data_item in f.readlines() if "<source>" in data_item]
        print(f"number of data records in AG-Corpus: {len(agcorpus_datalines)}")

    data_obj_dict = {}
    for data_item in agcorpus_datalines:
        tmp_title = data_item[data_item.index("<title>") + len("<title>"): data_item.index("</title>")]
        tmp_category = data_item[data_item.index("<category>") + len("<category>"): data_item.index("</category>")]
        tmp_description = data_item[data_item.index("<description>") + len("<description>"): data_item.index("</description>")]

        tmp_agcorpus_obj = AGCorpus(name="agcorpus", topic=tmp_category, title=tmp_title, description=tmp_description)
        if tmp_category not in data_obj_dict.keys():
            data_obj_dict[tmp_category] = [tmp_agcorpus_obj]
        else:
            data_obj_dict[tmp_category].append(tmp_agcorpus_obj)
    print("#" * 10)
    print("> check data in AG-Corpus :")
    for data_key, data_value in data_obj_dict.items():
        print(f"\t {data_key}: {len(data_value)}")
    print("#" * 10)
    return data_obj_dict

def load_agnews_data(data_dir, num_data_in_dev_per_label=1900):
    label_file = os.path.join(data_dir, "classes.txt")
    with open(label_file, "r") as f:
        idx_to_label_dict = {str(label_idx + 1): label_item.strip() for label_idx, label_item in
                             enumerate(f.readlines())}

    train_file_path = os.path.join(data_dir, "train.csv")
    test_file_path = os.path.join(data_dir, "test.csv")
    train_dev_obj_dict = {}
    train_obj_dict = {}
    dev_obj_dict = {}
    test_obj_dict = {}
    # class_idx, question title, question content and best answer

    with open(train_file_path, "r") as train_csv_f:
        train_datareader = csv.reader(train_csv_f, delimiter=',')
        for train_data_item in train_datareader:
            data_content = train_data_item
            label_content = idx_to_label_dict[data_content[0]]
            tmp_data_record = AGNews(name="agnews", data_type="train", topic=label_content, title=data_content[1], description=data_content[2])
            if label_content not in train_dev_obj_dict.keys():
                train_dev_obj_dict[label_content] = [tmp_data_record]
            else:
                train_dev_obj_dict[label_content].append(tmp_data_record)

    for key_item in train_dev_obj_dict.keys():
        num_obj = len(train_dev_obj_dict[key_item])
        num_obj_in_dev = num_data_in_dev_per_label
        obj_index_lst = [idx for idx in range(0, num_obj)]
        obj_idx_for_dev = random.sample(obj_index_lst, num_obj_in_dev)
        obj_idx_for_train = list(set(obj_index_lst) - set(obj_idx_for_dev))
        dev_obj_dict[key_item] = [train_dev_obj_dict[key_item][idx] for idx in obj_idx_for_dev]
        train_obj_dict[key_item] = [train_dev_obj_dict[key_item][idx] for idx in obj_idx_for_train]

    with open(test_file_path, "r") as test_csv_f:
        test_datareader = csv.reader(test_csv_f, delimiter=',')
        for test_data_item in test_datareader:
            data_content = test_data_item
            label_content = idx_to_label_dict[data_content[0]]
            tmp_data_record = AGNews(name="agnews", data_type="test", topic=label_content, title=data_content[1], description=data_content[2])
            if label_content not in test_obj_dict.keys():
                test_obj_dict[label_content] = [tmp_data_record]
            else:
                test_obj_dict[label_content].append(tmp_data_record)
    print("#"*10)
    print("> check train in AGNews :")
    for train_key, train_value in train_obj_dict.items():
        print(f"\t {train_key}: {len(train_value)}")
    print("> check dev in AGNews :")
    for dev_key, dev_value in dev_obj_dict.items():
        print(f"\t {dev_key}: {len(dev_value)}")
    print("> check test in AGNews :")
    for test_key, test_value in test_obj_dict.items():
        print(f"\t {test_key}: {len(test_value)}")
    print("#"*10)
    return train_obj_dict, dev_obj_dict, test_obj_dict

def get_id_and_ood_data_statistic():
    data_label = {
        "ood_test_per_label": 1900,
        "ood_test": 7600,
        "id_test_per_label": 1900,
        "id_test": 7600,
        "ood_dev_per_label": 1900,
        "ood_dev": 7600,
        "id_dev_per_label": 1900,
        "id_dev": 7600,
        "train_per_label": 28100,
        "train":  112400
    }
    return data_label

def split_id_ood_distribution_strategy(agnews_train_dict, agnews_dev_dict, agnews_test_dict, agcorpus_data_dict):
    data_statistic_dict = get_id_and_ood_data_statistic()

    id_topic_agnews = ["World", "Sports", "Business", "Sci/Tech"]
    ood_topic_agcorpus = ["World", "Sports", "Business", "Sci/Tech"]

    print(f">>> BEFORE removing duplicate data pairs : ")
    for tmp_topic in id_topic_agnews:
        print(f"AG-Corpus {tmp_topic} -> {len(agcorpus_data_dict[tmp_topic])}")

    remove_agcorpus_idx_dict = {}
    for ood_topic in ood_topic_agcorpus:
        print(f"REMOVE {ood_topic}")
        remove_agcorpus_idx_lst = []
        for data_item in agnews_train_dict[ood_topic]:
            data_item_desc = data_item.description
            find_bool = 0
            for candi_data_idx, candi_data_item in enumerate(agcorpus_data_dict[ood_topic]):
                if find_bool == 0 and candi_data_item.description != data_item_desc:
                    continue
                elif find_bool == 0 and candi_data_item.description == data_item_desc:
                    remove_agcorpus_idx_lst.append(candi_data_idx)
                    find_bool = 1
                elif find_bool == 1:
                    continue
                else:
                    ValueError
        print(f"## AGCORPUS TRAIN DONE. ")

        for data_item in agnews_dev_dict[ood_topic]:
            data_item_desc = data_item.description
            find_bool = 0
            for candi_data_idx, candi_data_item in enumerate(agcorpus_data_dict[ood_topic]):
                if find_bool == 0 and candi_data_item.description != data_item_desc:
                    continue
                elif find_bool == 0 and candi_data_item.description == data_item_desc:
                    remove_agcorpus_idx_lst.append(candi_data_idx)
                    find_bool = 1
                elif find_bool == 1:
                    continue
                else:
                    ValueError
        print(f"## AGCORPUS DEV DONE. ")

        for data_item in agnews_test_dict[ood_topic]:
            data_item_desc = data_item.description
            find_bool = 0
            for candi_data_idx, candi_data_item in enumerate(agcorpus_data_dict[ood_topic]):
                if find_bool == 0 and candi_data_item.description != data_item_desc:
                    continue
                elif find_bool == 0 and candi_data_item.description == data_item_desc:
                    remove_agcorpus_idx_lst.append(candi_data_idx)
                    find_bool = 1
                elif find_bool == 1:
                    continue
                else:
                    ValueError
        print(f"CHECK REMOVE ...")
        print(len(set(remove_agcorpus_idx_lst)))
        remove_agcorpus_idx_dict[ood_topic] = set(remove_agcorpus_idx_lst)
        print(f"## AGCORPUS TEST DONE. ")

    print(f">>> AFTER removing duplicate data pairs : ")
    for tmp_topic in ood_topic_agcorpus:
        print(f"AG-Corpus {tmp_topic} -> {len(remove_agcorpus_idx_dict[tmp_topic])}")

    train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict = {}, {}, {}, {}, {}

    for id_topic_item in id_topic_agnews:
        train_in_dist_dict[id_topic_item] = agnews_train_dict[id_topic_item]
        dev_in_dist_dict[id_topic_item] = agnews_dev_dict[id_topic_item]
        test_in_dist_dict[id_topic_item] = agnews_test_dict[id_topic_item]

    for ood_topic_item in ood_topic_agcorpus:
        num_obj_in_dev = data_statistic_dict["ood_dev_per_label"]
        num_obj_in_test = data_statistic_dict["ood_test_per_label"]

        obj_index_lst = [idx for idx in range(len(agcorpus_data_dict[ood_topic_item]))]
        obj_index_lst = list(set(obj_index_lst) - set(remove_agcorpus_idx_dict[ood_topic_item]))
        ood_dev_obj_index = random.sample(obj_index_lst, num_obj_in_dev)
        left_obj_idx_lst = list(set(obj_index_lst) - set(ood_dev_obj_index))
        ood_test_obj_index = random.sample(left_obj_idx_lst, num_obj_in_test)

        dev_out_of_dist_dict[ood_topic_item] = [agcorpus_data_dict[ood_topic_item][idx] for idx in ood_dev_obj_index]
        test_out_of_dist_dict[ood_topic_item] = [agcorpus_data_dict[ood_topic_item][idx] for idx in ood_test_obj_index]

    return train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict

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

def get_argument_parser():
    parser = argparse.ArgumentParser(description="return agnews-ext argument parser.")
    parser.add_argument("--ag_news_data_dir", type=str, required=True, help="yahoo_answers data dir")
    parser.add_argument("--ag_corpus_data_file", type=str, required=True, help="ag_corpus data dir")
    parser.add_argument("--save_data_dir", type=str, required=True, help="")
    parser.add_argument("--save_original_data_to_csv", action="store_true")

    return parser

def main():
    parser = get_argument_parser()
    input_arguments = parser.parse_args()

    agnews_train_dict, agnews_dev_dict, agnews_test_dict = load_agnews_data(input_arguments.ag_news_data_dir)
    agcorpus_data_dict = load_agcorpus_data(input_arguments.ag_corpus_data_file)
    train_id_dict, dev_id_dict, test_id_dict, dev_ood_dict, test_ood_dict = split_id_ood_distribution_strategy(agnews_train_dict, agnews_dev_dict, agnews_test_dict, agcorpus_data_dict)

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