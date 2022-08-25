from typing import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ...utils.helpers import str2regex
from .base import BaseFeaturizer


class StatsFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.char_counter = CountVectorizer(
            tokenizer=lambda x: list(x), lowercase=False, binary=True
        )
        self.regex_counter = CountVectorizer(
            tokenizer=lambda x: [x], lowercase=False, binary=True
        )
        self.regex_counter2 = CountVectorizer(
            tokenizer=lambda x: [x], lowercase=False, binary=True
        )

    def fit(self, dirty_df: pd.DataFrame, col: str):
        self.char_counter.fit(dirty_df[col].values.tolist())
        self.regex_counter.fit(
            [str2regex(val, match_whole_token=False) for val in dirty_df[col].values]
        )

        self.regex_counter2.fit(
            [str2regex(val, match_whole_token=True) for val in dirty_df[col].values]
        )

    def transform(self, dirty_df: pd.DataFrame, col: str):
        char_features = self.char_counter.transform(
            dirty_df[col].values.tolist()
        ).todense()

        regex_features = self.regex_counter.transform(
            [str2regex(val, match_whole_token=False) for val in dirty_df[col].values]
        ).todense()

        regex_features2 = self.regex_counter2.transform(
            [str2regex(val, match_whole_token=True) for val in dirty_df[col].values]
        ).todense()


        return np.concatenate([char_features, regex_features, regex_features2], axis=1)

    def n_features(self, dirty_df: pd.DataFrame):
        return sum(
            [
                len(x.get_feature_names())
                for x in [self.char_counter, self.regex_counter, self.regex_counter2]
            ]
        )

    def feature_dim(self):
        return 1


class TfidfFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: list(x), ngram_range=(1, 2))
        self.sym_tfidf = TfidfVectorizer(tokenizer=lambda x: list(x))

    def fit(self, dirty_df: pd.DataFrame, col: str):
        self.tfidf.fit(dirty_df[col].values.tolist())
        self.sym_tfidf.fit(
            dirty_df[col].apply(lambda x: str2regex(x, match_whole_token=False)).values
        )

    def transform(self, dirty_df: pd.DataFrame, col: str):
        tfidf = self.tfidf.transform(dirty_df[col].values.tolist()).todense()

        sym_tfidf = self.sym_tfidf.transform(
            dirty_df[col].apply(lambda x: str2regex(x, match_whole_token=False)).values
        ).todense()

        return np.concatenate([tfidf], axis=1)

    def n_features(self, dirty_df: pd.DataFrame):
        return sum([len(x.get_feature_names()) for x in [self.tfidf]])

    def feature_dim(self):
        return 1
