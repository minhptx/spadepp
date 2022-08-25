from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from sklearn.linear_model import LogisticRegression
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.decomposition import PCA


class TripleLossSentTransformer():
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.clf = LogisticRegression()

    def reset(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def fit(self, dirty_df, labels, col):
        examples = [InputExample(texts=[x], label=y) for x, y in zip(dirty_df.loc[:, col].tolist(), labels)]

        train_dataset = SentencesDataset(examples, self.model)

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
        train_loss = losses.BatchAllTripletLoss(model=self.model)
        
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)
        train_features = self.model.encode(dirty_df.loc[:, col].values.tolist())
        self.clf.fit(train_features, labels)
        print(self.clf.score(train_features, labels))

    def predict(self, dirty_df, col):
        test_features = self.model.encode(dirty_df.loc[:, col].values.tolist())
        return self.clf.predict_proba(test_features)[:, 1]