#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: yahoo_agnews_five.py

import os
import csv
import random
import argparse
import collections
from utils.random_seed import set_random_seed
set_random_seed(2333)
from data.data_utils.clean_text import clean_20newsgroup_data, tokenize_and_clean_text_str


AGCorpus = collections.namedtuple("AGCorpus",["name", "topic", "title", "description"])
YahooAnswers = collections.namedtuple("YahooAnswers",["name", "data_type", "topic", "question_title", "question_content", "best_answer"])

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

def load_yahoo_answers_data(data_dir, dev_ratio=0.03):
    label_file = os.path.join(data_dir, "classes.txt")
    with open(label_file, "r") as f:
        idx_to_label_dict = {str(label_idx+1): label_item.strip() for label_idx, label_item in enumerate(f.readlines())}

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
            tmp_data_record = YahooAnswers(name="yahooanswers", data_type="train", topic=label_content, question_title=data_content[1],
                                           question_content=data_content[2], best_answer=data_content[3])
            if label_content not in train_dev_obj_dict.keys():
                train_dev_obj_dict[label_content] = [tmp_data_record]
            else:
                train_dev_obj_dict[label_content].append(tmp_data_record)

    for key_item in train_dev_obj_dict.keys():
        num_obj_in_train = len(train_dev_obj_dict[key_item])
        num_obj_in_dev = int(dev_ratio * num_obj_in_train)
        obj_index_lst = [idx for idx in range(0, num_obj_in_train)]
        obj_idx_for_dev = random.sample(obj_index_lst, num_obj_in_dev)
        obj_idx_for_train = list(set(obj_index_lst) - set(obj_idx_for_dev))
        dev_obj_dict[key_item] = [train_dev_obj_dict[key_item][idx] for idx in obj_idx_for_dev]
        train_obj_dict[key_item] = [train_dev_obj_dict[key_item][idx] for idx in obj_idx_for_train]

    with open(test_file_path, "r") as test_csv_f:
        test_datareader = csv.reader(test_csv_f, delimiter=',')
        for test_data_item in test_datareader:
            data_content = test_data_item
            label_content = idx_to_label_dict[data_content[0]]
            tmp_data_record = YahooAnswers(name="yahooanswers", data_type="test", topic=label_content, question_title=data_content[1],
                                           question_content=data_content[2], best_answer=data_content[3])
            if label_content not in test_obj_dict.keys():
                test_obj_dict[label_content] = [tmp_data_record]
            else:
                test_obj_dict[label_content].append(tmp_data_record)

    return train_obj_dict, dev_obj_dict, test_obj_dict

def get_id_and_ood_data_statistic():
    data_label = {
        "ood_test_per_label": 500,
        "ood_test": 2500,
        "id_test_per_label": 500,
        "id_test": 2500,
        "ood_dev_per_label": 500,
        "ood_dev": 2500,
        "id_dev_per_label": 500,
        "id_dev": 2500,
        "train_per_label": 135000,
        "train":  675000
    }
    return data_label

def split_id_ood_distribution_strategy(yahoo_train_obj_dict, yahoo_dev_obj_dict, yahoo_test_obj_dict, agcorpus_obj_dict):
    data_statistic_dict = get_id_and_ood_data_statistic()

    id_topic_yahoo = ["Health", "Science & Mathematics", "Sports", "Entertainment & Music", "Business & Finance"]
    # ID-yahoo: "Health", "Science & Mathematics", "Sports", "Entertainment & Music", "Business & Finance"

    ood_topic_agnews = ["Health", "Sci/Tech", "Sports", "Entertainment", "Business"]
    # OOD-agnews: "Health", "Sci/Tech", "Sports", "Entertainment", "Business"
    train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict = {}, {}, {}, {}, {}

    for id_topic_idx, id_topic_item in enumerate(id_topic_yahoo):
        num_train = data_statistic_dict["train_per_label"]
        train_candidate_index_lst = [idx for idx in range(len(yahoo_train_obj_dict[id_topic_item]))]
        train_idx_lst = random.sample(train_candidate_index_lst, num_train)

        num_id_dev = data_statistic_dict["id_dev_per_label"]
        id_dev_candidate_index_lst = [idx for idx in range(len(yahoo_dev_obj_dict[id_topic_item]))]
        dev_idx_lst = random.sample(id_dev_candidate_index_lst, num_id_dev)

        num_id_test = data_statistic_dict["id_test_per_label"]
        id_test_candidate_index_lst = [idx for idx in range(len(yahoo_test_obj_dict[id_topic_item]))]
        test_idx_lst = random.sample(id_test_candidate_index_lst, num_id_test)

        train_in_dist_dict[id_topic_item] = [yahoo_train_obj_dict[id_topic_item][idx] for idx in train_idx_lst]
        dev_in_dist_dict[id_topic_item] = [yahoo_dev_obj_dict[id_topic_item][idx] for idx in dev_idx_lst]
        test_in_dist_dict[id_topic_item] = [yahoo_test_obj_dict[id_topic_item][idx] for idx in test_idx_lst]

    for ood_topic_idx, ood_topic_item in enumerate(ood_topic_agnews):
        ood_index_lst = [idx for idx in range(len(agcorpus_obj_dict[ood_topic_item]))]
        num_ood_dev = data_statistic_dict["ood_dev_per_label"]
        num_ood_test = data_statistic_dict["ood_test_per_label"]
        dev_idx_lst = random.sample(ood_index_lst, num_ood_dev)
        test_candidate_ood_index_lst = list(set(ood_index_lst) - set(dev_idx_lst))
        test_idx_lst = random.sample(test_candidate_ood_index_lst, num_ood_test)
        dev_out_of_dist_dict[ood_topic_item] = [agcorpus_obj_dict[ood_topic_item][idx] for idx in dev_idx_lst]
        test_out_of_dist_dict[ood_topic_item] = [agcorpus_obj_dict[ood_topic_item][idx] for idx in test_idx_lst]

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
                if data_item.name == "agcorpus":
                    text_content = data_item.title + " " + data_item.description
                elif data_item.name == "yahooanswers":
                    text_content = data_item.question_title + " " + data_item.question_content
                else:
                    raise ValueError
                cleaned_data = tokenize_and_clean_text_str(text_content)
                writer.writerow({'label': data_obj_key,
                                 'data': repr(cleaned_data),})

    print("*="*10)
    print(f">>> save file to : {data_file_path}")
    print(f">>> the number of record is : {data_counter}")

def get_argument_parser():
    parser = argparse.ArgumentParser(description="return yahoo_agnews_five argument parser.")
    parser.add_argument("--yahoo_answers_data_dir", type=str, required=True, help="yahoo_answers data dir")
    parser.add_argument("--ag_corpus_data_file", type=str, required=True, help="ag_corpus data dir")
    parser.add_argument("--save_data_dir", type=str, required=True, help="")
    parser.add_argument("--save_original_data_to_csv", action="store_true")

    return parser

def main():
    parser = get_argument_parser()
    input_arguments = parser.parse_args()

    yahoo_train_obj_dict, yahoo_dev_obj_dict, yahoo_test_obj_dict = load_yahoo_answers_data(input_arguments.yahoo_answers_data_dir)
    agcorpus_obj_dict = load_agcorpus_data(input_arguments.ag_corpus_data_file)

    train_id_dict, dev_id_dict, test_id_dict, dev_ood_dict, test_ood_dict = split_id_ood_distribution_strategy(yahoo_train_obj_dict, yahoo_dev_obj_dict, yahoo_test_obj_dict, agcorpus_obj_dict)

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