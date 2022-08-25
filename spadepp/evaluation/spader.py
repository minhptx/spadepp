import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

from spader.models.xgboost import XGBoost

from ..models.random_forest import RandomForest
from ..models.lr import LR

from ..models.sentence_transformer import TripleLossSentTransformer
from ..models.transformer import Transformer
from ..models.fcn import FCN
from ..labeler import Oracle

from ..detection.detector import Detector

from ..data_source import DataContainer


class SpaderEvaluation:
    def __init__(self) -> None:
        pass

    def evaluate(self, data_path, num_examples=20, output_to_file=None, k=4):
        dataset = DataContainer.from_path(data_path, dtype=str, keep_default_na=False)
        # detector = Detector(TripleLossSentTransformer(), Oracle(dataset))
        # detector = Detector(Transformer(), Oracle(dataset))
        # detector = Detector(FCN(), Oracle(dataset))
        # detector = Detector(RandomForest(), Oracle(dataset))

        interval = num_examples // k

        Path(output_to_file).mkdir(parents=True, exist_ok=True)

        for num_example in range(0, num_examples + 1, interval):
            # detector = Detector(XGBoost(), Oracle(dataset))
            detector = Detector(XGBoost(), Oracle(dataset))


            if output_to_file:
                output_file = Path(output_to_file) / f"{num_example}.txt"
                writer = open(output_file, "w")
            else:
                writer = sys.stdout

            predictions, output = detector.detect(dataset, num_example)

            for col in dataset.columns:
                output.col2report[col] = pd.DataFrame(classification_report(dataset.groundtruth_df[col], predictions[col], output_dict=True)).transpose().to_markdown()

            output.report = pd.DataFrame(classification_report(dataset.groundtruth_df.values.flatten(), predictions.values.flatten(), output_dict=True)).transpose().to_markdown()
            output.to_file(writer)
