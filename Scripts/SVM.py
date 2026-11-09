# %% Import / Youness
from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from Pipeline import pipeline


# %% Scoring pour la sélection des paramètres
def precision_top20(estimator, X, y):
    scores = estimator.decision_function(X)
    k = max(1, int(np.ceil(0.20 * len(y))))
    selection = np.argsort(-scores, kind="stable")[:k]

    return float(np.mean(np.asarray(y)[selection] == 1))


cv_strategy = TimeSeriesSplit(n_splits=5)


# %% SVM RBF — Recherche des paramètres
param_grid_rbf = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__gamma": [0.01, 0.1, 1, 10]
}

model_rbf = GridSearchCV(
    estimator=pipeline(SVC(kernel="rbf", class_weight="balanced")),
    param_grid=param_grid_rbf,
    cv=cv_strategy,
    scoring=precision_top20,
    verbose=2

)



# %% SVM polynomial — Recherche des paramètres
param_grid_poly = {
    "model__degree": [2, 3, 4],
    "model__coef0": [10, 20]
}

model_poly = GridSearchCV(
    estimator=pipeline(SVC(kernel="poly", class_weight="balanced")),
    param_grid=param_grid_poly,
    cv=cv_strategy,
    scoring=precision_top20,
    verbose=2

)
