#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: process_word_emb.py

import os
import numpy as np


def extract_glove_vocab_and_weights(glove_embedding_file_path, save_dir):
    """
    $ head -n 1 glove.6B.50d.txt
    > the 0.418 0.24968 -0.41242 0.1217 0.34527 -0.044457 -0.49688 -0.17862 -0.00066023 -0.6566 0.27843 -0.14767 -0.55677 0.14658 -0.0095095 0.011658 0.10204 -0.12792 -0.8443 -0.12181 -0.016801 -0.33279 -0.1552 -0.23131 -0.19181 -1.8823 -0.76746 0.099051 -0.42125 -0.19526 4.0071 -0.18594 -0.52287 -0.31681 0.00059213 0.0074449 0.17778 -0.15897 0.012041 -0.054223 -0.29871 -0.15749 -0.34758 -0.045637 -0.44251 0.18785 0.0027849 -0.18411 -0.11514 -0.78581
    """
    vec_dim_size = int(glove_embedding_file_path.split(".")[-2].replace("d", "")) # 300d,glove.6B.300d.txt
    with open(glove_embedding_file_path, "r") as f:
        embedding_tables = f.readlines()

    token_lst = []
    weight_lst = []
    for word_embedding in embedding_tables:
        word_and_embedding_weight = word_embedding.strip().split(" ")
        word_token = word_and_embedding_weight[0]
        word_weight = [float(value) for value in word_and_embedding_weight[1:]]
        token_lst.append(word_token)
        weight_lst.append(word_weight)

    # <PAD>
    token_lst.insert(0, "<PAD>")
    pad_vec = np.random.uniform(-0.25, 0.25, vec_dim_size)
    weight_lst.insert(0, pad_vec)

    # <UNK>
    token_lst.insert(1, "<UNK>")
    unk_vec = np.random.uniform(-0.25, 0.25, vec_dim_size)
    weight_lst.insert(1, unk_vec)

    # save vocab file and np.weight
    glove_npy_file = os.path.join(save_dir, glove_embedding_file_path.split("/")[-1].replace(".txt", ".npy"))
    with open(glove_npy_file, "wb") as f:
        weight_npy = np.array(weight_lst)
        np.save(f, weight_npy)
        print(f"INFO -> save {glove_npy_file} ")

    vocab_file = os.path.join(save_dir, glove_embedding_file_path.split("/")[-1].replace(".txt", f"_vocab_{len(token_lst)}.txt"))
    with open(vocab_file, "w") as f:
        for token in token_lst:
            f.write(f"{token}\n")
        print(f"INFO -> save {vocab_file}")



if __name__ == "__main__":
    glove_embedding_file_path = "/data/nfsdata/nlp/embeddings/glove/glove.6B.300d.txt"
    save_dir = "/data/xiaoya/datasets/confidence/embeddings"
    extract_glove_vocab_and_weights(glove_embedding_file_path, save_dir)