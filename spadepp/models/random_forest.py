import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, optim
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
from accelerate import Accelerator

from .features.statistics import StatsFeaturizer, TfidfFeaturizer
from .features.base import UnionFeaturizer
from datasets import Dataset


class RandomForest():
    def __init__(self):
        self.feature_extractor = StatsFeaturizer()

        self.model = RandomForestClassifier(min_samples_split=10)

    def extract_features(self, dirty_df, labels, col, training=True):
        if training:
            features = self.feature_extractor.fit_transform(dirty_df, col)
        else:
            features = self.feature_extractor.transform(dirty_df, col)

        if labels is not None:
            return {"features": np.concatenate(features), "labels": labels}
        return {"features": np.concatenate(features)}

    def reset(self):
        try:
            self.model = RandomForestClassifier()
            self.feature_extractor = StatsFeaturizer()
        except:
            pass

    def fit(self, dirty_df, labels, col):
        data_dict = self.extract_features( 
            dirty_df, labels, col
        )

        self.model.fit(data_dict["features"], data_dict["labels"])


    def predict(self, dirty_df, col):
        data_dict = self.extract_features(
            dirty_df, None, col, training=False
        )

        return self.model.predict_proba(data_dict["features"])[:, 1]