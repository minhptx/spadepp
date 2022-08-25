from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import FastText
from spellchecker import SpellChecker


class UnarySignal(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame, col):
        pass

    def transform(self, dirty_df: pd.DataFrame, col):
        pass

    def fit_transform(self, dirty_df: pd.DataFrame, col, **kwargs):
        self.fit(dirty_df, col)

        return self.transform(dirty_df, col, **kwargs)


class BinarySignal(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, dirty_df: pd.DataFrame):
        pass

    def transform(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        pass

    def fit_transform(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        self.fit(dirty_df)

        return self.transform(dirty_df, lhs_col, rhs_col)

    def fit_transform_metal(self, dirty_df: pd.DataFrame, lhs_col: str, rhs_col: str):
        scores = self.fit_transform(dirty_df, lhs_col, rhs_col)
        val_arr = np.ones((len(scores), 1)) * -1

        indices = np.where(scores == np.min(scores))[0]

        val_arr[indices] = 0.0

        return val_arr


class SignalUtils:
    instance = None
    tokenizer = None
    spell_checker = None
    bigram_dict = None

    @staticmethod
    def get_instance():
        if SignalUtils.instance is None:
            SignalUtils.instance = FastText()
        return SignalUtils.instance

    @staticmethod
    def get_tokenizer():
        if SignalUtils.tokenizer is None:
            SignalUtils.tokenizer = get_tokenizer("spacy")
        return SignalUtils.tokenizer

    @staticmethod
    def get_spell_checker():
        if SignalUtils.spell_checker is None:
            SignalUtils.spell_checker = SpellChecker()
        return SignalUtils.spell_checker

    @staticmethod
    def get_bigram_dict():
        if SignalUtils.bigram_dict is None:
            df = pd.read_csv("resources/bigram_count.csv")
            SignalUtils.bigram_dict = df.set_index("data").to_dict()["count"]
        return SignalUtils.bigram_dict

