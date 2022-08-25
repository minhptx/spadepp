import random
import time

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import classification_report, accuracy_score

from spader.utils.debugger import Debugger, FakeDebugger

from ..semi_supervised.active_learner import LRLearner, MaxUncertaintyLearner, MaxErrorLearner, MixedLearner, ProbLearner
from ..semi_supervised.data_augmentation.error_generator import ErrorGenerator
from ..semi_supervised.data_augmentation.label_propagation import LabelPropagator
from .signalizer import Signalizer
from loguru import logger
import os
from dataclasses import dataclass, field

@dataclass
class Output:
    report: Dict = field(default_factory=dict)
    propagation_report: Dict = field(default_factory=dict)
    running_time: int = 0
    col2report: Dict = field(default_factory=dict)
    num_propagated_positive: int = 0
    num_propagated_negative: int = 0


    def to_file(self, f):
        f.write(f"Overall report:\n{self.report}\n")
        f.write(f"Running time: {self.running_time}\n")
        f.write("-------------------------------------\n")
        f.write(f"Propagation report:\n{self.propagation_report}\n")
        f.write(f"Number of propagated positive examples: {self.num_propagated_positive}\n")
        f.write(f"Number of propagated negative examples: {self.num_propagated_negative}\n")
        f.write("-------------------------------------")
        for col, report in self.col2report.items():
            f.write(f"Detail report for column {col}:\n")
            f.write(str(report) + "\n")
            f.write("--------------------------------------\n")


class Detector:
    def __init__(self, model, labeler):
        self.model = model
        self.output = Output()
        self.propagator = LabelPropagator(self.output)
        self.active_learner = LRLearner(self.propagator)
        self.error_generator = ErrorGenerator()
        self.signalizer = Signalizer()
        self.labeler = labeler

    def reset(self):
        self.model.reset()
        self.active_learner.reset()
        self.error_generator.reset()
        self.signalizer = Signalizer()
        self.output = Output()

    def detect(self, dataset, n_samples, debug=True):
        if debug:
            debugger = Debugger(os.path.join("debug", dataset.name))
        else:
            debugger = FakeDebugger()

        predictions = pd.DataFrame(
            -1.0, columns=dataset.dirty_df.columns, index=dataset.dirty_df.index
        )

        propagated_labels = []
        groundtruths = []

        for col in dataset.columns:
            self.reset()
            logger.info("Column: {}".format(col))

            col_labeler, col_corrector = self.labeler.get_labeler_for_col(col), self.labeler.get_correct_for_col(col)
            generated_df, labels, rules, original_labels = self.preprocess(dataset, col, n_samples, col_labeler, col_corrector, debugger)

            if labels is not None:
                propagated_labels.extend([-1 if x not in [0,1] else x for x in original_labels])
            else:
                propagated_labels.extend([-1 for _ in range(len(dataset.dirty_df))])

            groundtruths.extend(dataset.groundtruth_df[col].values.tolist())

            if generated_df is None:
                logger.info("No negative examples found for column {}".format(col))
                predictions[col] = True
                continue

            debugger.debug_series(f"{col}_generated_data", generated_df[col], labels, rules)

            self.model.fit(generated_df, labels, col)
            predictions[col] = self.model.predict(dataset.dirty_df, col)
            debugger.debug_series(f"{col}_predictions", dataset.dirty_df[col], predictions[col], dataset.groundtruth_df[col])

            predictions[col] = predictions[col] > 0.5

        return predictions, self.output

    
    def preprocess(self, dataset, col, n_samples, col_labeler, col_corrector, debugger=FakeDebugger()):
        features = self.signalizer.fit_transform(dataset.dirty_df, col)
        values, labels = dataset.get_data(col)


        start_time = time.time()
        pos_indices, neg_indices, labels, preds = self.active_learner.query(features, labels, n_samples, col_labeler, values, debugger, col)
        logger.info("Labeled {} rows".format(len(pos_indices) + len(neg_indices)))

        self.output.running_time = (time.time() - start_time) /  20

        self.output.propagation_report = classification_report([x if labels[i] != np.nan else -1 for i, x in enumerate(dataset.groundtruth_df[col])], labels.fillna(-1), output_dict=True)
        self.output.num_propagated_positive += len([x for x in labels if x == 1]) - len(pos_indices)
        self.output.num_propagated_negative +=  len([x for x in labels if x == 0]) - len(neg_indices)

        error_values = values[neg_indices].values.tolist()
        corrected_values = [col_corrector(neg_index) for neg_index in neg_indices]

        debugger.debug_series(f"{col}_chosen", error_values + values[pos_indices].values.tolist(), corrected_values + values[pos_indices].values.tolist())

        if len(neg_indices) == 0:
            return None, None, None, None

        raw2transformed = self.error_generator.fit_transform(corrected_values, error_values, values)
        logger.info("Learned transformations")

        generated_df = pd.DataFrame(columns=dataset.columns)
        new_labels = []
        new_rules = []
        count_generated = 0

        for idx, row in dataset.dirty_df.iterrows():
            if labels[idx] == 0 or preds[idx] < 0.15:
                generated_df.loc[len(generated_df)] = row
                new_labels.append(0)
                new_rules.append(preds[idx])
            elif labels[idx] == 1 or preds[idx] > 0.85:
                generated_df.loc[len(generated_df)] = row
                new_labels.append(1)

                if row[col] in raw2transformed:
                    transformed_value = random.choice(list(raw2transformed[row[col]]))
                    new_row = row.copy()
                    new_row[col] = transformed_value
                    generated_df.loc[len(generated_df)] = new_row
                    new_labels.append(0)
                    count_generated += 1
                    new_rules.append(f"Generated from {row[col]}")

        for index, corrected_value in zip(neg_indices, corrected_values):
            new_row = dataset.dirty_df.loc[index].copy()
            new_row[col] = corrected_value
            generated_df.loc[len(generated_df)] = new_row
            new_labels.append(1)
            new_rules.append(f"Corrected from {values[index]}")

        logger.info("Generated {} examples for column {}".format(count_generated, col))
        debugger.debug_str("Generated {} examples for column {}".format(count_generated, col))

        return generated_df, new_labels, new_rules, labels

    


            