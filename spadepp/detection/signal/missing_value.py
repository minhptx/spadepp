import numpy as np
import pandas as pd

from spader.detection.signal.base import UnarySignal

class MissingValueSignal(UnarySignal):
    def __init__(self):
        pass

    def fit(self, dirty_df: pd.DataFrame, col):
        pass

    def transform(self, dirty_df: pd.DataFrame, col):
        return dirty_df[col].apply(lambda x: 0 if x.strip().lower() in ["", "null", "none", "na"] else 1).values