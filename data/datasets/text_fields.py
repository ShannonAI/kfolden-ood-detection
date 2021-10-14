#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: text_fields.py

from collections import namedtuple


# text fields for semantic shift datasets
AGNewsFLDataExample = namedtuple("AGNewsFLDataExample", [])
AGNewsFMDataExample = namedtuple("AGNewsFMDataExample", [])
ReutersmKnLDataExample = namedtuple("ReutersmKnLDataExample", [])
YahooAnswersFMDataExample = namedtuple("YahooAnswersFMDataExample", [])


# text fields for nonsemantic shift datasets
AGNewsExtDataExample = namedtuple("AGNewsExtDataExample", [])
TwentyNewsSixSDataExample = namedtuple("TwentyNewsSixSDataExample", [])
YahooAGNewsFiveDataExample = namedtuple("YahooAGNewsFiveDataExample", [])


non_semantic_shift_fields = {
    "agnews_ext": AGNewsExtDataExample,
    "twenty_news_sixs": TwentyNewsSixSDataExample,
    "yahoo_agnews_five": YahooAGNewsFiveDataExample
}

semantic_shift_fields = {
    "agnews_fl": AGNewsFLDataExample,
    "agnews_fm": AGNewsFMDataExample,
    "reuters_mk_nl": ReutersmKnLDataExample,
    "yahoo_answers_fm": YahooAnswersFMDataExample
}

