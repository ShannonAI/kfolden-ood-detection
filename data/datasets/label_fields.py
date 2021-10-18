#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: label_fields.py


def get_labels(data_sign, dist_sign="id"):
    if data_sign == "agnews_ext":
        if dist_sign == "id":
            return []
        return []
    elif data_sign == "20news_6s":
        if dist_sign == "id":
            return ["comp", "rec", "sci", "religion", "politics", "misc"]
        return ["comp", "rec", "sci", "religion", "politics", "misc"]
    elif data_sign == "yahoo_agnews_five":
        if dist_sign == "id":
            return ["Health", "Science & Mathematics", "Sports", "Entertainment & Music", "Business & Finance"]
        return ["Health", "Sci/Tech", "Sports", "Entertainment", "Business"]
    elif data_sign == "agnews_fl":
        if dist_sign == "id":
            return ["World", "Sports", "Business", "Sci/Tech"]
        return ["U.S.", "Europe", "Italia", "Software and Developement"]
    elif data_sign == "agnews_fm":
        if dist_sign == "id":
            return ["World", "Sports", "Business", "Sci/Tech"]
        return ["Entertainment", "Health", "Top Stories", "Music Feeds"]
    elif data_sign == "yahoo_answers_fm":
        if dist_sign == "id":
            return ["Health", "Science & Mathematics", "Sports", "Entertainment & Music", "Business & Finance"]
        return ["Society & Culture", "Education & Reference", "Computers & Internet", "Family & Relationships", "Politics & Government"]

    return [0, 1]
