import numpy as np
from sklearn.model_selection import GridSearchCV , TimeSeriesSplit
from Pipeline import pipeline
from SVM import precision_top20




def score_logistique(X, theta):
    n, _ = np.shape(X)
    X_aug = np.column_stack((X, np.ones((n,))))
    return X_aug @ theta  

def proba(z):
    return 1 / (1 + np.exp(-z)) 





def cout_logistique(X, y, theta):
    p = proba(score_logistique(X, theta))

    p = np.clip(p, 1e-12, 1 - 1e-12) 

    J = -y * np.log(p) - (1 - y) * np.log(1 - p) 
    return np.mean(J)

def gradient_logistique(X, y, theta):
    n, d = np.shape(X)
    X = np.array(X)
    X_aug = np.column_stack((X, np.ones((n))))
    z = score_logistique(X, theta)
    p = proba(z)

    dJ = X_aug.T @ (p - y)
    return 1 / n * dJ

def entrainer_logistique(X, y, n_iterations, alpha):
    n, d = np.shape(X)
    X = np.array(X)

    X_aug = np.column_stack((X, np.ones((n))))
    theta = np.zeros((d + 1,))
    couts = []

    for i in range(n_iterations):
        theta -= alpha * gradient_logistique(X, y, theta)
        couts.append(cout_logistique(X, y, theta))

    return theta, couts

def predict_logistique(X, theta, s):
    z = score_logistique(X, theta)
    p = proba(z)
    return (p >= s).astype(int)



from sklearn.base import BaseEstimator, ClassifierMixin


class Logistique(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.01, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations

    def _preparer_X(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=float)

    def fit(self, X, y):
        X = self._preparer_X(X)
        y = np.asarray(y)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Les reponses d'entrainement doivent contenir 0 et 1.")

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        self.theta_, self.couts_ = entrainer_logistique(
            X, y, self.n_iterations, self.alpha
        )
        return self

    def decision_function(self, X):
        X = self._preparer_X(X)
        return score_logistique(X, self.theta_)

    def predict_proba(self, X):
        p = proba(self.decision_function(X))
        return np.column_stack((1 - p, p))

    def predict(self, X):
        scores = self.decision_function(X)
        m = max(1, int(np.ceil(0.20 * len(scores))))
        selection = np.argsort(-scores, kind="stable")[:m]
        y_pred = np.zeros(len(scores), dtype=int)
        y_pred[selection] = 1
        return y_pred

    def score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "La fonction de cout ne gere pas les poids des observations."
            )
        X = self._preparer_X(X)
        # GridSearchCV maximise ce score : on renvoie donc l'oppose du cout.
        return -float(cout_logistique(X, np.asarray(y), self.theta_))


cv_strategy = TimeSeriesSplit(n_splits = 5)

param_grid_logistique = {
    "model__alpha" : [0.005,0.01,0.1,0.5],
    "model__n_iterations" : [500,1000,2500,5000]
}

model_logistique_base = GridSearchCV(
    estimator=pipeline(Logistique()),
    param_grid=param_grid_logistique,
    cv=cv_strategy,
    n_jobs=-1,
    verbose=2,
    scoring = precision_top20
)

