from .signal import (
    CharFormatSignal,
    DictTypoSignal,
    FastTextSignal,
    MissingValueSignal,
    PunctFormatSignal,
    WebTableBoolSignal,
    CharValueSignal,
    WordFormatSignal,
    MinCharFrequencySignal
)

import numpy as np

class Signalizer:
    def __init__(self) -> None:
        self.name2signal = [
                (f"missing_values", MissingValueSignal()),
                (f"fasttext", FastTextSignal()),
                (f"dict_typo", DictTypoSignal()),
                (f"min_char", MinCharFrequencySignal()),
                (f"char", CharValueSignal()),
                ("web", WebTableBoolSignal()),
                (f"char_format", CharFormatSignal()),
                (f"word_format", WordFormatSignal()),
        ]


    def fit_transform(self, df, col):
        return np.concatenate([
            signal.fit_transform(df, col).reshape(-1, 1)
            for name, signal in self.name2signal
        ], axis=1)