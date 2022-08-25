import numpy as np
import pandas as pd

from loguru import logger

class LabelPropagator:
    def __init__(self, output, level = 0.01):
        self.level = level
        self.output = output

    def fit(self, features, labels, indices, values, debugger, col):
        feature_strs = pd.DataFrame(features).applymap(lambda x: str(self.level * np.round(x / self.level))).agg("|".join, axis=1)
        
        feature2label = {}
        feature2idx = {}
        for index in indices:
            feature2label[feature_strs[index]] = labels[index]
            feature2idx[feature_strs[index]] = index

        reasons = []
        propagated_count = 0
        for index, feature in enumerate(feature_strs):
            if index not in indices and feature in feature2label:
                labels[index] = feature2label[feature]
                propagated_count += 1
                reasons.append("Propagated from value {}".format(values[feature2idx[feature]]))
            else:
                reasons.append("No propagated label for feature {}".format(feature))


        debugger.debug_series(f"{col}_propagated", values.values.tolist(), reasons, labels, feature_strs.values.tolist())
        self.output.num_propagation = len([x for x in labels if x in [0, 1]]) - len(indices)

        return labels
    