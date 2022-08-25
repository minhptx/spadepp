from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from sklearn.linear_model import LogisticRegression
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

class Transformer():
    def __init__(self):
        self.model = BertForSequenceClassification(BertConfig())
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def reset(self):
        self.model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)

    def tokenize(self, examples):
        encodings = self.tokenizer(examples["data"], truncation=True, padding=True)
        if "label" in examples:
            encodings["labels"] = np.array(examples["label"]).astype(np.int64)
        return encodings

    def softmax(self, x):
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    def fit(self, dirty_df, labels, col):
        dataset = Dataset.from_dict({"data": dirty_df.loc[:, col].values.tolist(), "label": labels})
        dataset = dataset.map(self.tokenize, batched=True, remove_columns=["label", "data"])

        training_args = TrainingArguments(
            output_dir='./output',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            seed=0,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        self.trainer.train()

    def predict(self, dirty_df, col):
        dataset = Dataset.from_dict({"data": dirty_df.loc[:, col].values.tolist()})
        dataset = dataset.map(self.tokenize, batched=True)        
        predictions = self.trainer.predict(dataset).predictions
        return self.softmax(predictions)[:, 1]


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
        train_loss = losses.BatchHardSoftMarginTripletLoss(model=self.model)
        
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)
        train_features = self.model.encode(dirty_df.loc[:, col].values.tolist())
        self.clf = LogisticRegression()
        self.clf.fit(train_features, labels)

    def predict(self, dirty_df, col):
        test_features = self.model.encode(dirty_df.loc[:, col].values.tolist())
        return self.clf.predict(test_features)