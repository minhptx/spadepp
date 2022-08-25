from pathlib import Path

import pandas as pd
from skactiveml.utils import MISSING_LABEL


class DataContainer:
    def __init__(self, name, dirty_df, clean_df):
        self.name = name
        self.dirty_df = dirty_df
        self.clean_df = clean_df

        self.groundtruth_df = self.clean_df == self.dirty_df
        self.label_df = pd.DataFrame(
            MISSING_LABEL, columns=self.dirty_df.columns, index=self.dirty_df.index
        )

    @property
    def columns(self):
        return self.dirty_df.columns

    def get_data(self, col):
        return self.dirty_df.loc[:, col], self.label_df.loc[:, col]

    @staticmethod
    def from_path(file_path, **kwargs):
        file_path = Path(file_path)
        clean_df = pd.read_csv(
            file_path / "clean.csv", **kwargs
        )
        dirty_df = pd.read_csv(
            file_path / "dirty.csv", **kwargs
        )

        # dirty_df = dirty_df.loc[:, ["ibu"]]
        # clean_df = clean_df.loc[:, ["ibu"]]

        return DataContainer(file_path.name, dirty_df, clean_df)
