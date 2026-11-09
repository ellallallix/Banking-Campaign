# %% Imports — aucune lecture du jeu de test
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from joblib import load


# %% Comparaisons appariees, sur les vrais oui des memes clients
def comparer_selections(y, choix, reference):
    y, choix, reference = np.asarray(y), np.asarray(choix), np.asarray(reference)
    if y.shape != choix.shape or y.shape != reference.shape:
        raise ValueError("Les deux selections doivent correspondre aux memes clients.")
    k = int(choix.sum())
    if k == 0 or int(reference.sum()) != k:
        raise ValueError("Les deux modeles doivent selectionner le meme nombre de clients.")
    positifs = y == 1
    m = int(positifs.sum())
    if m == 0:
        raise ValueError("La comparaison du rappel necessite au moins un vrai oui.")

    # d = +1 si seul le choix retrouve ce oui, -1 si seule la reference le retrouve.
    d = choix[positifs].astype(float) - reference[positifs].astype(float)
    gagnes, perdus = int((d == 1).sum()), int((d == -1).sum())
    moyenne = float(d.mean())
    ecart_type = float(d.std(ddof=1)) if m > 1 else 0.0
    erreur_standard = ecart_type / np.sqrt(m) * m / k
    gain_precision = (gagnes - perdus) / k
    p_normale = (float(norm.sf(gain_precision / erreur_standard))
                 if erreur_standard > 0 else (1.0 if moyenne == 0 else None))
    # Version unilaterale du test exact de McNemar sur la capture des vrais oui.
    p_exacte = (float(binomtest(gagnes, gagnes + perdus, .5, alternative="greater").pvalue)
               if gagnes + perdus else 1.0)
    marge = float(norm.ppf(.975) * erreur_standard)
    return {
        "clients": len(y), "oui": m, "selectionnes": k,
        "oui_retrouves": int((y * choix).sum()),
        "oui_reference": int((y * reference).sum()),
        "precision_choix": float((y * choix).sum() / k),
        "precision_reference": float((y * reference).sum() / k),
        "gagnes": gagnes, "perdus": perdus,
        "moyenne_d": moyenne, "ecart_type_d": ecart_type,
        "gain_rappel": moyenne, "gain_precision": gain_precision,
        "erreur_standard_gain_precision": float(erreur_standard),
        "ic95_bas": gain_precision - marge, "ic95_haut": gain_precision + marge,
        "p_normale_approx": p_normale, "p_brute": p_exacte,
    }


def corriger_holm(p_values):
    p = np.asarray(p_values, dtype=float)
    ordre = np.argsort(p, kind="stable")
    corrigees = np.empty_like(p)
    corrigees[ordre] = np.minimum(1, np.maximum.accumulate(p[ordre] * (len(p) - np.arange(len(p)))))
    return corrigees


# %% Analyse de validation des modeles deja entraines, sans fit
def analyser_validation(
    fichier_initial="modeles_entraines.joblib",
    fichier_simplifie="modeles_entraines_simplifies.joblib",
):
    from Data import X_val, y_val
    from EntrainementSimplifie import simplifier_donnees
    from Evaluation import predictions_top20

    initial, simplifie = load(fichier_initial), load(fichier_simplifie)
    X_simple, y_simple = simplifier_donnees(X_val, y_val)
    # Conserver pdays et day_of_week en entree du modele initial, sur les memes lignes.
    X_initial_commun = X_val.loc[X_simple.index]
    noms = {
        "svm_rbf": "SVM RBF", "svm_poly": "SVM polynomial",
        "logistique": "Logistique classique", "logistique_adaptee": "Logistique adaptee",
    }
    lignes = []
    # Chaque modele est compare au modele aleatoire deja defini.
    for version, resultats, X, y in (
        ("Initiale", initial, X_val, y_val),
        ("Simplifiee", simplifie, X_simple, y_simple),
    ):
        reference = predictions_top20(resultats["aleatoire"]["modele"], X, aleatoire=True)
        for cle, nom in noms.items():
            choix = predictions_top20(resultats[cle]["modele"], X)
            lignes.append({
                "version": version, "modele": nom, "reference": "Aleatoire",
                "test": "McNemar unilateral sur vrais oui",
                **comparer_selections(y, choix, reference),
            })

    # Seule comparaison entre versions : la logistique adaptee sur les memes clients.
    choix = predictions_top20(simplifie["logistique_adaptee"]["modele"], X_simple)
    reference = predictions_top20(initial["logistique_adaptee"]["modele"], X_initial_commun)
    lignes.append({
        "version": "Comparaison des versions", "modele": "Logistique adaptee simplifiee",
        "reference": "Logistique adaptee initiale", "test": "McNemar unilateral sur vrais oui",
        **comparer_selections(y_simple, choix, reference),
    })

    tableau = pd.DataFrame(lignes)
    tableau["p_holm"] = corriger_holm(tableau["p_brute"])
    tableau["gain_detecte_5pct"] = (tableau["gain_precision"] > 0) & (tableau["p_holm"] < .05)
    affichage = tableau[["version", "modele", "reference", "gain_precision", "p_brute", "p_holm"]].copy()
    for col in ("p_brute", "p_holm"):
        affichage[col] = affichage[col].map(lambda p: f"{100*p:.4g} %")
    affichage["gain_precision"] = affichage["gain_precision"].map(lambda g: f"{100*g:+.2f} points")
    print(affichage.to_string(index=False))
    return tableau


if __name__ == "__main__":
    resultats = analyser_validation()
