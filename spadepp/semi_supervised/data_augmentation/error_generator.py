from .transformation_learner import TransformationLearner
import collections

class ErrorGenerator:
    def __init__(self) -> None:
       self.transformation_learner = TransformationLearner() 

    def reset(self):
        self.transformation_learner = TransformationLearner() 


    def fit(self, cleaned_values, error_values, all_values):
        transform_ops = self.transformation_learner.fit(cleaned_values, error_values)
        return transform_ops


    def transform(self, cleaned_values, error_values, all_values):
        raw2transformed = collections.defaultdict(set)

        for value in all_values:
            for transform_op in self.transformation_learner.transform_ops:
                if transform_op.validate(value):
                    new_value = transform_op.transform(value)
                    if new_value not in raw2transformed[value]:
                        raw2transformed[value].add(new_value)

        return raw2transformed


    def fit_transform(self, cleaned_values, error_values, all_values_with_indices):
        self.fit(cleaned_values, error_values, all_values_with_indices)
        new_values = self.transform(cleaned_values, error_values, all_values_with_indices)

        return new_values
        


