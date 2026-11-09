
# %% Donnees simplifiees et entrainement
from pathlib import Path
import numpy as np
from Data import X_train, y_train
from Entrainement import entrainer_modeles


COLONNES_EXCLUES = ("pdays", "day_of_week")


def simplifier_donnees(X, y):
    # Filtrer dans chaque periode existante, sans refaire le decoupage chronologique.
    masque = ~X.eq("unknown").any(axis=1)
    X_simple = X.loc[masque].drop(columns=list(COLONNES_EXCLUES)).copy()
    y_simple = np.asarray(y)[masque.to_numpy()]
    return X_simple, y_simple


def entrainer_version_simplifiee(fichier_sortie=None):
    if fichier_sortie is None:
        fichier_sortie = Path(__file__).resolve().parent / "modeles_entraines_simplifies.joblib"
    X_simple, y_simple = simplifier_donnees(X_train, y_train)
    print(f"Entrainement simplifie : {len(X_train)} -> {len(X_simple)} clients.")
    return entrainer_modeles(
        fichier_sortie, X_train=X_simple, y_train=y_simple,
        colonnes_exclues=COLONNES_EXCLUES,
    )


if __name__ == "__main__":
    resultats = entrainer_version_simplifiee()



