import numpy as np
from random import sample, seed
from sklearn.base import BaseEstimator, ClassifierMixin
from Pipeline import pipeline


def predict_random(X):
    n, _ = np.shape(X)
    m = int(np.ceil(0.2 * n))
    y_pred = [0] * n

    seed(42)
    r = sample(range(n), m)

    for i in r:
        y_pred[i] = 1

    return y_pred


class ModeleAleatoire(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.shape(X)[1]
        return self

    def predict(self, X):
        return np.array(predict_random(X))


model_aleatoire = pipeline(ModeleAleatoire())
