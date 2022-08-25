import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
from accelerate import Accelerator

from .features.statistics import StatsFeaturizer
from .features.base import UnionFeaturizer
from datasets import Dataset, tqdm
from sentence_transformers import SentenceTransformer

class Reducer(nn.Module):
    def __init__(self, input_dim, reduce_dim):
        super(Reducer, self).__init__()

        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.model = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2), nn.ReLU(), nn.Linear(input_dim, reduce_dim),
        )

    def forward(self, inputs):
        if inputs.shape[0] != 1:
            inputs = self.batch_norm(inputs)
        return self.model(inputs)


class FCNModule(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()

        self.reducers = nn.ModuleList()

        for feature_dim in feature_dims:
            self.reducers.append(Reducer(feature_dim, 10))

        self.batch_norm = nn.BatchNorm1d(
            10 * len(feature_dims)
        )

        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(10 * len(feature_dims), 1,),
        )

    def forward(self, *features):
        concat_inputs = []

        for idx, inputs in enumerate(features):
            concat_inputs.append(self.reducers[idx](inputs.float()))

        attn_outputs = torch.cat(concat_inputs, dim=1)

        if attn_outputs.shape[0] != 1:
            attn_outputs = self.batch_norm(attn_outputs)

        return torch.sigmoid(self.model(attn_outputs.float()))


class FCN():
    def __init__(self):
        self.feature_extractor = UnionFeaturizer(
            {"stats": StatsFeaturizer()}
        )

        self.feature_dims = []
        self.encoder = SentenceTransformer("all-MiniLM-L12-v2")
        self.accelerator = Accelerator()

    def extract_features(self, dirty_df, labels, col, training=True):
        if training:
            features = self.feature_extractor.fit_transform(dirty_df, col)
        else:
            features = self.feature_extractor.transform(dirty_df, col)

        embeddings = self.encoder.encode(dirty_df[col].values.tolist())

        self.feature_dims = [
            extractor.n_features(dirty_df)
            for extractor in self.feature_extractor.name2extractor.values()
        ] + [embeddings.shape[1]]

        if labels is not None:
            return {"features": np.concatenate(features), "labels": labels, "embeddings": embeddings}
        return {"features": np.concatenate(features), "embeddings": embeddings}

    def reset(self):
        try:
            self.model = FCNModule(self.feature_dims)
        except Exception as e:
            print(e)

    def fit(self, dirty_df, labels, col):
        data_dict = self.extract_features(
            dirty_df, labels, col
        )

        self.reset()

        self.model.train()

        dataset = Dataset.from_dict(data_dict)
        dataset.set_format(type="torch", columns=["features", "labels", "embeddings"])

        optimizer = torch.optim.Adam(self.model.parameters())

        data = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)
        self.model, optimizer, data = self.accelerator.prepare(self.model, optimizer, data)

        for _ in range(5):
            for batch in data:
                optimizer.zero_grad()
                output = self.model(batch["features"], batch["embeddings"])
                loss = F.binary_cross_entropy_with_logits(output, batch["labels"].unsqueeze(1))
                self.accelerator.backward(loss)
                optimizer.step()


    def predict(self, dirty_df, col):
        data_dict = self.extract_features(
            dirty_df, None, col, training=False
        )

        dataset = Dataset.from_dict(data_dict)
        dataset.set_format(type="torch", columns=["features", "embeddings"])

        data = torch.utils.data.DataLoader(dataset, batch_size=32)

        data = self.accelerator.prepare(data)

        self.model.eval()

        predictions = []
        for batch in data:
            predictions.extend(self.model(batch["features"], batch["embeddings"])[:, 0].detach().cpu().numpy().tolist())

        return np.asarray(predictions)
