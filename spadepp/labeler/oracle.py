class Oracle:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def get_correct_for_col(self, col):
        return lambda idx: self.dataset.clean_df[col][idx]

    def get_labeler_for_col(self, col):
        def get_label(idx):
            return self.dataset.groundtruth_df[col][idx]
        return get_label