import numpy as np
from random import sample, seed as fixer_seed
from sklearn.base import BaseEstimator, ClassifierMixin
from Pipeline import pipeline


def predict_random(X, seed=42):
    n, _ = np.shape(X)
    m = int(np.ceil(0.2 * n))
    y_pred = [0] * n

    fixer_seed(seed)
    r = sample(range(n), m)

    for i in r:
        y_pred[i] = 1

    return y_pred


def evaluer_aleatoire(y, graines=range(100)):
    """Moyenne et ecart-type des scores de tirages gardant tous le meme quota."""
    y = np.asarray(y)
    graines = tuple(graines)
    if len(graines) < 2 or len(set(graines)) != len(graines):
        raise ValueError("Utiliser au moins deux graines differentes.")
    if len(y) == 0:
        raise ValueError("L'evaluation necessite au moins un client.")
    # Le tirage ne depend que du nombre de lignes, pas des caracteristiques.
    X = np.empty((len(y), 0))
    k = int(np.ceil(.2 * len(y)))
    retrouves = np.array([np.dot(y, predict_random(X, seed=s)) for s in graines])
    precisions = retrouves / k
    rappels = retrouves / y.sum() if y.sum() else np.zeros(len(graines))
    return {
        "graines": graines, "nb_tirages": len(graines), "selectionnes": k,
        "precision_moyenne": float(precisions.mean()),
        "precision_ecart_type": float(precisions.std(ddof=1)),
        "rappel_moyen": float(rappels.mean()),
        "rappel_ecart_type": float(rappels.std(ddof=1)),
    }


class ModeleAleatoire(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = np.shape(X)[1]
        return self

    def predict(self, X, seed=42):
        return np.array(predict_random(X, seed=seed))


model_aleatoire = pipeline(ModeleAleatoire())
