#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: clean_text.py

import re
import os

def tokenize_and_clean_text_str(string, TREC=False, lower_case=False):
    """
    code from: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    # Roberta is case-sensitive.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if lower_case:
        return string.strip() if TREC else string.strip().lower()
    else:
        return string.strip() if TREC else string.strip()

def clean_20newsgroup_data(data_item, remove=["headers", "footers", "quotes"], merge_doc=True):
    if "headers" in remove:
        data_item = strip_newsgroup_header(data_item)
    if "footers" in remove:
        data_item = strip_newsgroup_footer(data_item)
    if "quotes" in remove:
        data_item = strip_newsgroup_quoting(data_item)
    if merge_doc:
        data_item = data_item.replace("\n", " ")
    return data_item

def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    source: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f1541b1ca4570177d93fd7979613497e3/sklearn/datasets/twenty_newsgroups.py#L100
    """
    _before, _blankline, after = text.partition('\n\n')
    return after

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                       r'|^In article|^Quoted from|^\||^>)')

def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    source: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f1541b1ca4570177d93fd7979613497e3/sklearn/datasets/twenty_newsgroups.py#L113
    """
    good_lines = [line for line in text.split('\n') if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)

def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    source: https://github.com/scikit-learn/scikit-learn/blob/f0ab589f1541b1ca4570177d93fd7979613497e3/sklearn/datasets/twenty_newsgroups.py#L124
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text

def remove_stop_and_lowfreq_words_func(doc_str_lst, stopwords_file=None, freq_threshold=5):
    word2freq_dict = {}
    doc_tokens_lst = [doc_item.split(" ") for doc_item in doc_str_lst]
    # doc_tokens_lst
    for doc_item in doc_tokens_lst:
        for token in doc_item:
            if token in word2freq_dict.keys():
                word2freq_dict[token] += 1
            else:
                word2freq_dict[token] = 1

    if stopwords_file is None:
        os.system("python3 -m nltk.downloader stopwords")
        # save to ~/nltk_data/corpora/stopwords/english
    elif not os.path.exists(stopwords_file):
        os.system("python3 -m nltk.downloader stopwords")
        # save to ~/nltk_data/corpora/stopwords/english
    else:
        with open(stopwords_file, "r") as f:
            stopwords = [word.strip() for word in f.readlines()]

    cleaned_doc_tokens_lst = []
    for doc_item in doc_tokens_lst:
        clean_doc_item = []
        for token in doc_item:
            if token not in stopwords and word2freq_dict[token] > freq_threshold:
                clean_doc_item.append(token)
        cleaned_doc_tokens_lst.append(clean_doc_item)

    cleaned_doc_str_lst = [" ".join(doc_item) for doc_item in cleaned_doc_tokens_lst]
    return cleaned_doc_str_lst, word2freq_dict
