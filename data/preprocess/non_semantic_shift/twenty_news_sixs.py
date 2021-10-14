#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: non_semantic_shift/twenty_news_sixs.py

import os
import csv
import argparse
import collections
from glob import glob
from utils.random_seed import set_random_seed
set_random_seed(2333)

TwentyNews = collections.namedtuple("TwentyNews",["data_type", "root_topic", "topic", "data", "file_name"])


def split_id_ood_distribution_strategy(train_obj_dict, dev_obj_dict, test_obj_dict):
    id_topic_map = {
        "comp.graphics": "comp",
        "comp.sys.ibm.pc.hardware": "comp",
        "comp.os.ms-windows.misc": "comp",
        "rec.autos": "rec",
        "rec.motorcycles": "rec",
        "sci.crypt": "sci",
        "sci.electronics": "sci",
        "talk.religion.misc": "religion",
        "talk.politics.guns": "politics",
        "talk.politics.misc": "politics",
        "misc.forsale": "misc"
    }
    ood_topic_map = {
        "comp.sys.mac.hardware": "comp",
        "comp.windows.x": "comp",
        "rec.sport.baseball": "rec",
        "rec.sport.hockey": "rec",
        "sci.med": "sci",
        "sci.space": "sci",
        "alt.atheism": "religion",
        "soc.religion.christian": "religion",
        "talk.politics.mideast": "politics"
    }

    train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict = {}, {}, {}, {}, {}
    for topic_item in id_topic_map.keys():
        merge_topic = id_topic_map[topic_item]
        if merge_topic not in train_in_dist_dict.keys():
            train_in_dist_dict[merge_topic] = train_obj_dict[topic_item]
        else:
            train_in_dist_dict[merge_topic].extend(train_obj_dict[topic_item])

        if merge_topic not in dev_in_dist_dict.keys():
            dev_in_dist_dict[merge_topic] = dev_obj_dict[topic_item]
        else:
            dev_in_dist_dict[merge_topic].extend(dev_obj_dict[topic_item])

        if merge_topic not in test_in_dist_dict.keys():
            test_in_dist_dict[merge_topic] = test_obj_dict[topic_item]
        else:
            test_in_dist_dict[merge_topic].extend(test_obj_dict[topic_item])

    for topic_item in ood_topic_map.keys():
        merge_topic = ood_topic_map[topic_item]
        if merge_topic not in dev_out_of_dist_dict.keys():
            dev_out_of_dist_dict[merge_topic] = dev_obj_dict[topic_item]
        else:
            dev_out_of_dist_dict[merge_topic].extend(dev_obj_dict[topic_item])

        if merge_topic not in test_out_of_dist_dict.keys():
            test_out_of_dist_dict[merge_topic] = test_obj_dict[topic_item]
        else:
            test_out_of_dist_dict[merge_topic].extend(test_obj_dict[topic_item])

    return train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict

def save_id_ood_data(save_data_dir, data_file_name, data_object_dict):
    # train_dir, dev_id_dir, test_id_dir, dev_ood_dir, test_ood_dir
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    data_file_path = os.path.join(save_data_dir, data_file_name)
    data_counter = 0
    with open(data_file_path, mode="w", encoding="utf-8", newline="\n") as w_csv_file:
        fieldnames = ['merge_label', 'data', 'label', ]
        writer = csv.DictWriter(w_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data_obj_key in data_object_dict.keys():
            for data_item in data_object_dict[data_obj_key]:
                data_counter += 1
                writer.writerow({'merge_label': data_obj_key,
                                 'data': repr(data_item.data),
                                 'label': data_item.topic,})

    print("*="*10)
    print(f">>> save file to : {data_file_path}")
    print(f">>> the number of record is : {data_counter}")

def get_argument_parser():
    parser = argparse.ArgumentParser(description="return 20news-6s argument parser.")
    parser.add_argument("--twentynews_data_dir", type=str, required=True, help="data dir")
    parser.add_argument("--save_data_dir", type=str, required=True, help="")
    parser.add_argument("--save_original_data_to_csv", action="store_true")

    return parser

def load_data(file_path):
    # return a <str>
    with open(file_path, "r") as f:
        data = f.read()
        return data

def return_label_from_file_path(file_path):
    # example: /data/lixiaoya/datasets/confidence/20news/dev/alt.atheism/51160
    file_infos_lst = file_path.split("/")
    file_name = file_infos_lst[-1] # 51160
    file_topic = file_infos_lst[-2] # alt.atheism
    file_type = file_infos_lst[-3] # dev
    file_root_topic = file_topic.split(".")[0]
    return file_name, file_topic, file_root_topic, file_type

def load_twenty_news_data(data_dir, save_csv_data=False):
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")
    test_data_dir = os.path.join(data_dir, "test")

    train_file_lst = glob(os.path.join(train_data_dir, "*/*"))
    dev_file_lst = glob(os.path.join(dev_data_dir, "*/*"))
    test_file_lst = glob(os.path.join(test_data_dir, "*/*"))

    train_object_dict = {}
    dev_object_dict = {}
    test_object_dict = {}

    for train_file in train_file_lst:
        tmp_file_name, tmp_file_topic, tmp_file_root_topic, tmp_file_type = return_label_from_file_path(train_file)
        tmp_data = load_data(train_file)
        tmp_train_data_obj = TwentyNews(data_type=tmp_file_type, topic=tmp_file_topic, root_topic=tmp_file_root_topic, data=tmp_data, file_name=tmp_file_name)
        if tmp_file_topic not in train_object_dict.keys():
            train_object_dict[tmp_file_topic] = [tmp_train_data_obj]
        else:
            train_object_dict[tmp_file_topic].append(tmp_train_data_obj)
    print(f"number of topic in train : {len(train_object_dict.keys())}")

    for dev_file in dev_file_lst:
        tmp_file_name, tmp_file_topic, tmp_file_root_topic, tmp_file_type = return_label_from_file_path(dev_file)
        tmp_data = load_data(dev_file)
        tmp_dev_data_obj = TwentyNews(data_type=tmp_file_type, topic=tmp_file_topic, root_topic=tmp_file_root_topic, data=tmp_data, file_name=tmp_file_name)
        if tmp_file_topic not in dev_object_dict.keys():
            dev_object_dict[tmp_file_topic] = [tmp_dev_data_obj]
        else:
            dev_object_dict[tmp_file_topic].append(tmp_dev_data_obj)
    print(f"number of topic in dev: {len(dev_object_dict.keys())}")

    for test_file in test_file_lst:
        tmp_file_name, tmp_file_topic, tmp_file_root_topic, tmp_file_type = return_label_from_file_path(test_file)
        tmp_data = load_data(test_file)
        tmp_test_data_obj = TwentyNews(data_type=tmp_file_type, topic=tmp_file_topic, root_topic=tmp_file_root_topic, data=tmp_data, file_name=tmp_file_name)
        if tmp_file_topic not in test_object_dict.keys():
            test_object_dict[tmp_file_topic] = [tmp_test_data_obj]
        else:
            test_object_dict[tmp_file_topic].append(tmp_test_data_obj)
    print(f"number of topic in test: {len(test_object_dict.keys())}")

    print("#" * 10)
    print("> check train in 20News :")
    for train_key, train_value in train_object_dict.items():
        print(f"\t {train_key}: {len(train_value)}")
    print("> check dev in 20News :")
    for dev_key, dev_value in dev_object_dict.items():
        print(f"\t {dev_key}: {len(dev_value)}")
    print("> check test in 20News :")
    for test_key, test_value in test_object_dict.items():
        print(f"\t {test_key}: {len(test_value)}")
    print("#" * 10)

    return train_object_dict, dev_object_dict, test_object_dict


def main():
    parser = get_argument_parser()
    input_arguments = parser.parse_args()

    train_object_dict, dev_object_dict, test_object_dict = load_twenty_news_data(input_arguments.twentynews_data_dir, )
    train_in_dist_dict, dev_in_dist_dict, test_in_dist_dict, dev_out_of_dist_dict, test_out_of_dist_dict = split_id_ood_distribution_strategy(train_object_dict, dev_object_dict, test_object_dict)

    save_train_dir = os.path.join(input_arguments.save_data_dir, "train")
    save_dev_dir = os.path.join(input_arguments.save_data_dir, "dev")
    save_test_dir = os.path.join(input_arguments.save_data_dir, "test")

    print(f"$$$ TRAIN: ")
    save_id_ood_data(save_train_dir, "train.csv", train_in_dist_dict)
    print(f"$$$ ID DEV: ")
    save_id_ood_data(save_dev_dir, "id_dev.csv", dev_in_dist_dict)
    print(f"$$$ OOD DEV: ")
    save_id_ood_data(save_dev_dir, "ood_dev.csv", dev_out_of_dist_dict)
    print(f"$$$ ID TEST: ")
    save_id_ood_data(save_test_dir, "id_test.csv", test_in_dist_dict)
    print(f"$$$ OOD TEST: ")
    save_id_ood_data(save_test_dir, "ood_test.csv", test_out_of_dist_dict)


if __name__ == "__main__":
    main()


