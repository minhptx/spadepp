from .random_forest import RandomForest
from xgboost import XGBClassifier
from .features.statistics import StatsFeaturizer, TfidfFeaturizer


class XGBoost(RandomForest):
    def __init__(self):
        super().__init__()
        self.model = XGBClassifier()

    def reset(self):
        try:
            self.model = XGBClassifier()
            self.feature_extractor = StatsFeaturizer()
        except:
            pass
