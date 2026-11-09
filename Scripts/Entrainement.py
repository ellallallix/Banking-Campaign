
# %% Import / Youness
from Data import X_train, y_train
from SVM import model_rbf, model_poly
from LogisticRegression import model_logistique
from Random import model_aleatoire
from LogisticRegressionBase import model_logistique_base

import argparse
import time
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from joblib import dump


def selection_top20(scores):
    scores = np.asarray(scores)
    k = max(1, int(np.ceil(0.20 * len(scores))))
    selection = np.argsort(-scores, kind="stable")[:k]
    predictions = np.zeros(len(scores), dtype=int)
    predictions[selection] = 1
    return predictions


# %% Entrainement et conservation des sorties, sans evaluation
def entrainer_modeles(
    fichier_sortie="modeles_entraines.joblib",
    X_train=X_train, y_train=y_train, colonnes_exclues=(),
):
    resultats = {}

    # Les pipelines et les CV sont definies dans les scripts des modeles.
    for nom, definition in (
        ("svm_rbf", model_rbf),
        ("svm_poly", model_poly),
        ("logistique_adaptee", model_logistique),
        ("aleatoire", model_aleatoire),
        ("logistique", model_logistique_base),
    ):
        entrainement = clone(definition)
        avec_cv = isinstance(entrainement, GridSearchCV)
        if colonnes_exclues:
            # Reutiliser le pretraitement existant en retirant les colonnes exclues.
            modele_initial = entrainement.estimator if avec_cv else entrainement
            preprocess = modele_initial.named_steps["preprocess"]
            preprocess.set_params(transformers=[
                (nom_bloc, transformation,
                 [col for col in colonnes if col not in colonnes_exclues])
                for nom_bloc, transformation, colonnes in preprocess.transformers
            ])
        debut = time.perf_counter()
        entrainement.fit(X_train, y_train)
        duree = time.perf_counter() - debut

        modele = entrainement.best_estimator_ if avec_cv else entrainement
        estimateur = modele.named_steps["model"]

        if nom == "aleatoire":
            scores = None
            predictions = np.asarray(modele.predict(X_train), dtype=int)
        else:
            if hasattr(modele, "decision_function"):
                scores = np.asarray(modele.decision_function(X_train))
            else:
                scores = np.asarray(modele.predict_proba(X_train))[:, 1]
            predictions = selection_top20(scores)

        resultats[nom] = {
            "modele": modele,
            "parametres": estimateur.get_params(),
            "parametres_cv": entrainement.best_params_ if avec_cv else None,
            "resultats_cv": entrainement.cv_results_ if avec_cv else None,
            "score_cv": entrainement.best_score_ if avec_cv else None,
            "scoring_cv": entrainement.scoring if avec_cv else None,
            "scores_train": scores,
            "predictions_train": predictions,
            # Facultatifs : lus si ton estimateur expose ces attributs.
            "theta": getattr(estimateur, "theta_", None),
            "couts": getattr(estimateur, "couts_", None),
            "duree_cv_et_entrainement": duree,
        }
        print(f"{nom} : entrainement termine.")
        dump(resultats, fichier_sortie)

    dump(resultats, fichier_sortie)
    print(f"Modeles et sorties conserves dans : {fichier_sortie}")
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sortie", default="modeles_entraines.joblib")
    arguments = parser.parse_args()
    resultats = entrainer_modeles(arguments.sortie)



