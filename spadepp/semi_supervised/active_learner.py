import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from skactiveml.pool import UncertaintySampling 
from skactiveml.classifier import SklearnClassifier
from skactiveml.utils import unlabeled_indices, labeled_indices, MISSING_LABEL
from sklearn.utils.validation import check_is_fitted
from torch import logit_

def is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False


class ProbLearner:
    def __init__(self, label_propagator):
        self.label_propagator = label_propagator
        self.weights = None

    def reset(self):
        self.weights = None

    def query(self, X, y, n_samples, call_back, values=None, debugger=None, col=None):
        self.weights = np.ones((X.shape[1]))
        pos_indices = []
        neg_indices = []
        
        for i in range(n_samples):
            unlbld_idx = unlabeled_indices(y)

            if len(unlbld_idx) == 0:
                cand_idx = np.argmin(np.sum(np.matmul(X, self.weights)))
                y[cand_idx] = float(call_back(cand_idx))
            else:
                if len(pos_indices) > len(neg_indices):
                    cand_idx = np.argmin(np.sum(np.matmul(X[unlbld_idx], self.weights)))
                else:
                    print(np.matmul(X[unlbld_idx], self.weights).shape)

                    cand_idx = np.argmax(np.sum(np.matmul(X[unlbld_idx], self.weights)))
                cand_idx = unlbld_idx[cand_idx]
                y[cand_idx] = float(call_back(cand_idx))
            
            if y[cand_idx] == 1:
                pos_indices.append(cand_idx)
            else:
                neg_indices.append(cand_idx)


            y = self.label_propagator.fit(X, y, pos_indices + neg_indices, values, debugger, col)
        
        return pos_indices, neg_indices, y, y

class LRLearner:
    def __init__(self, label_propagator):
        self.clf = SklearnClassifier(LogisticRegression(), classes=[0, 1])
        self.label_propagator = label_propagator

    def reset(self):
        self.clf = SklearnClassifier(LogisticRegression(), classes=[0, 1])

    def query(self, X, y, n_samples, call_back, values=None, debugger=None, col=None):
        pos_indices = []
        neg_indices = []
        
        for i in range(n_samples):
            unlbld_idx = unlabeled_indices(y)

            if not is_fitted(self.clf):
                if len(unlbld_idx) == 0:
                    cand_idx = np.argmin(np.sum(X, axis=1))
                    y[cand_idx] = float(call_back(cand_idx))
                else:
                    cand_idx = np.argmin(np.sum(X[unlbld_idx], axis=1))
                    cand_idx = unlbld_idx[cand_idx]
                    y[cand_idx] = float(call_back(cand_idx))
            else:
                if len(unlbld_idx) == 0:
                    if len(pos_indices) > len(neg_indices):
                        cand_idx = np.argmin(self.clf.predict_proba(X)[:, 1])
                    else:
                        cand_idx = np.argmax(self.clf.predict_proba(X)[:, 1])
                    y[cand_idx] = float(call_back(cand_idx))
                else:
                    if len(pos_indices) > len(neg_indices):
                        cand_idx = np.argmin(self.clf.predict_proba(X[unlbld_idx])[:, 1])
                    else:
                        cand_idx = np.argmax(self.clf.predict_proba(X[unlbld_idx])[:, 1])
                    cand_idx = unlbld_idx[cand_idx]
                    y[cand_idx] = float(call_back(cand_idx))

            if y[cand_idx] == 1:
                pos_indices.append(cand_idx)
            else:
                neg_indices.append(cand_idx)

            self.clf.fit(X, y)

            y = self.label_propagator.fit(X, y, pos_indices + neg_indices, values, debugger, col)

        if is_fitted(self.clf):
            return pos_indices, neg_indices, y, self.clf.predict_proba(X)[:, 1]
        
        return pos_indices, neg_indices, y, y


class MaxUncertaintyLearner:
    def __init__(self, label_propagator, model=None):
        random_state = np.random.RandomState(0)
        if model is None:
            self.clf = SklearnClassifier(LogisticRegression(), classes=[0, 1])
        else:
            self.clf = SklearnClassifier(model, classes=[0, 1])
        self.qs = UncertaintySampling(method='entropy', random_state=random_state)
        self.label_propagator = label_propagator


    def reset(self):
        self.clf = SklearnClassifier(LogisticRegression(), classes=[0, 1])


    def query(self, X, y, n_samples, call_back, values=None, debugger=None, col=None):

        pos_indices = []
        neg_indices = []
        for i in range(n_samples):
            if i == 0:
                query_idx = np.argmin(np.sum(X, axis=1))
                y[query_idx] = float(call_back(query_idx))
                self.clf.fit(X, y)
            else:
                unlbld_idx = unlabeled_indices(y)
                if len(unlbld_idx) == 0:
                    X_cand = X
                    cand_idx = self.qs.query(X_cand=X_cand, clf=self.clf, batch_size=1)[0]
                    y[cand_idx] = float(call_back(cand_idx))
                else:
                    X_cand = X[unlbld_idx]
                    cand_idx = self.qs.query(X_cand=X_cand, clf=self.clf, batch_size=1)[0]
                    query_idx = unlbld_idx[cand_idx]
                    y[query_idx] = float(call_back(query_idx))
                self.clf.fit(X, y)
            
            if y[query_idx] == 1:
                pos_indices.append(query_idx)
            else:
                neg_indices.append(query_idx)

            y = self.label_propagator.fit(X, y, pos_indices + neg_indices, values, debugger, col)

        if is_fitted(self.clf):
            return pos_indices, neg_indices, y, self.clf.predict_proba(X)[:, 1]
        
        return pos_indices, neg_indices, y, y


class MaxErrorLearner:
    def __init__(self, label_propagator):
        self.clf = LogisticRegression()
        self.label_propagator = label_propagator

    def reset(self):
        self.clf = LogisticRegression()


    def query(self, X, y, n_samples, call_back, values=None, debugger=None, col=None, output=None):
        pos_indices = []
        neg_indices = []
        for i in range(n_samples):
            unlbld_idx = unlabeled_indices(y.values)

            if not is_fitted(self.clf):
                if len(unlbld_idx) == 0:
                    continue
                if False not in y.values:
                    query_idx = unlbld_idx[np.argmin(np.sum(X[unlbld_idx], axis=1))]
                    y[query_idx] = float(call_back(query_idx))
                else:
                    query_idx =  unlbld_idx[np.argmax(np.sum(X[unlbld_idx], axis=1))]
                    y[query_idx] = float(call_back(query_idx))
            else:
                if len(unlbld_idx) == 0:
                    unlbld_idx = set(range(len(y))).difference(set(pos_indices + neg_indices))
                    unlbld_idx = np.asarray(list(unlbld_idx)).astype(int)
                query_idx = unlbld_idx[np.argmax(self.clf.predict_proba(X[unlbld_idx])[:, 0])]
                y[query_idx] = float(call_back(query_idx))
            
            if False in y.values and True in y.values:
                lbld_idx = labeled_indices(y.values)
                self.clf.fit(X[lbld_idx], y[lbld_idx])
            
            if y[query_idx] == 1:
                pos_indices.append(query_idx)
            else:
                neg_indices.append(query_idx)
            y = self.label_propagator.fit(X, y, pos_indices + neg_indices, values, debugger, col)

        if is_fitted(self.clf):
            return pos_indices, neg_indices, y, self.clf.predict_proba(X)[:, 1]
        
        return pos_indices, neg_indices, y, y


class MixedLearner:
    def __init__(self, label_propagator) -> None:
        self.error_learner = MaxErrorLearner(label_propagator)
        self.uncertain_learner = MaxUncertaintyLearner(label_propagator, self.error_learner.clf)

    def reset(self):
        self.uncertain_learner.reset()
        self.error_learner.reset()

    def query(self, X, y, n_samples, call_back, values=None, debugger=None, col=None):
        error_pos_indices, error_neg_indices, y, error_probs = self.error_learner.query(X, y, n_samples // 2, call_back, values, debugger, col)
        uncertain_pos_indices, uncertain_neg_indices, y, uncertain_probs = self.uncertain_learner.query(X, y, n_samples // 2, call_back, values, debugger, col)
        pos_indices = uncertain_pos_indices + error_pos_indices
        neg_indices = uncertain_neg_indices + error_neg_indices
        return pos_indices, neg_indices, y, uncertain_probs