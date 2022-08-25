
from collections import defaultdict
import regex as re
import pandas as pd
from .base import UnarySignal
from ...utils.helpers import str2regex

class MinCharFrequencySignal(UnarySignal):
    def __init__(self):
        self.char_count = defaultdict(lambda: 0)

    def get_min_count(self, value):
        if not value:
            return 0
        tokens = list(value)
        if tokens:
            return min([self.char_count[x] for x in tokens])
        else:
            return self.char_count[""]

    def fit(self, dirty_df: pd.DataFrame, col):
        for value in dirty_df[col]:
            if not value:
                self.char_count[""] += 1
            for char in value:
                self.char_count[char] += 1

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].apply(lambda x: self.get_min_count(x) / len(dirty_df)).values


class CharValueSignal(UnarySignal):
    def __init__(self):
        super().__init__()
        self.counter = None

    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = dirty_df[col].value_counts().to_dict()

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].apply(lambda x: self.counter[x] / len(dirty_df)).values


class CharFormatSignal(CharValueSignal):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .apply(lambda x: str2regex(x, match_whole_token=False))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .apply(
                lambda x: self.counter[str2regex(x, match_whole_token=False)]
                / len(dirty_df)
            )
            .values
        )

class WordTypeSignal(CharValueSignal):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .apply(lambda x: "alphanum" if x.isalnum() else "alphabet" if x.isalpha() else "digit" if x.isdigit() else "other")
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .apply(
                lambda x: self.counter["alphanum" if x.isalnum() else "alphabet" if x.isalpha() else "digit" if x.isdigit() else "other"]
                / len(dirty_df)
            )
            .values
        )


class WordFormatSignal(CharValueSignal):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .apply(lambda x: str2regex(x, match_whole_token=True))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .apply(
                lambda x: self.counter[str2regex(x, match_whole_token=True)]
                / len(dirty_df)
            )
            .values
        )


class PunctFormatSignal(CharValueSignal):
    def fit(self, dirty_df: pd.DataFrame, col):
        self.counter = (
            dirty_df[col]
            .apply(lambda x: re.sub(r"[^\p{P}]+", "_", x))
            .value_counts()
            .to_dict()
        )

    def transform(self, dirty_df: pd.DataFrame, col):
        return (
            dirty_df[col]
            .apply(
                lambda x: self.counter[re.sub(r"[^\p{P}]+", "_", x)] / len(dirty_df)
            )
            .values
        )

