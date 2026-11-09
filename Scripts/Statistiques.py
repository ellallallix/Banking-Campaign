# %% Imports — aucune lecture du jeu de test
import numpy as np
import pandas as pd
from scipy.stats import norm
from joblib import load
from Random import evaluer_aleatoire


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
        "p_normale_approx": p_normale, "p_brute": p_normale,
    }


def comparer_aleatoire(y, choix, reference):
    """Gain moyen et SE sur les graines ; approximation normale deja utilisee."""
    y, choix = np.asarray(y), np.asarray(choix)
    k = int(choix.sum())
    if y.shape != choix.shape or k != reference["selectionnes"]:
        raise ValueError("Utiliser les memes clients et le meme quota que les tirages.")
    retrouves = int(np.dot(y, choix))
    precision = retrouves / k
    rappel = retrouves / y.sum() if y.sum() else 0.0
    gain = precision - reference["precision_moyenne"]
    # Chaque gain = precision du modele - precision du tirage.
    ecart_type = reference["precision_ecart_type"]
    se = ecart_type / np.sqrt(reference["nb_tirages"])
    p = (float(norm.sf(gain / se)) if se > 0 else
         (1.0 if gain == 0 else None))
    marge = float(norm.ppf(.975) * se)
    return {
        "clients": len(y), "oui": int(y.sum()), "selectionnes": k,
        "nb_tirages": reference["nb_tirages"], "oui_retrouves": retrouves,
        "precision_choix": precision, "precision_reference": reference["precision_moyenne"],
        "precision_reference_ecart_type": ecart_type,
        "rappel_reference": reference["rappel_moyen"],
        "rappel_reference_ecart_type": reference["rappel_ecart_type"],
        "gain_rappel": rappel - reference["rappel_moyen"], "gain_precision": gain,
        "ecart_type_gain_precision": ecart_type,
        "erreur_standard_gain_precision": float(se),
        "ic95_bas": gain - marge, "ic95_haut": gain + marge,
        "p_normale_approx": p, "p_brute": p,
    }


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
    # Chaque modele est compare a la moyenne des 100 tirages aleatoires.
    for version, resultats, X, y in (
        ("Initiale", initial, X_val, y_val),
        ("Simplifiee", simplifie, X_simple, y_simple),
    ):
        reference = evaluer_aleatoire(y)
        for cle, nom in noms.items():
            choix = predictions_top20(resultats[cle]["modele"], X)
            lignes.append({
                "version": version, "modele": nom, "reference": "Aleatoire moyen (100 graines)",
                "test": "Normale unilaterale, gain moyen / SE des tirages",
                **comparer_aleatoire(y, choix, reference),
            })

    # Seule comparaison entre versions : la logistique adaptee sur les memes clients.
    choix = predictions_top20(simplifie["logistique_adaptee"]["modele"], X_simple)
    reference = predictions_top20(initial["logistique_adaptee"]["modele"], X_initial_commun)
    lignes.append({
        "version": "Comparaison des versions", "modele": "Logistique adaptee simplifiee",
        "reference": "Logistique adaptee initiale", "test": "Normale unilaterale, gain / SE appariee",
        **comparer_selections(y_simple, choix, reference),
    })

    tableau = pd.DataFrame(lignes)
    tableau["gain_detecte_5pct"] = (tableau["gain_precision"] > 0) & (tableau["p_brute"] < .05)
    affichage = tableau[["version", "modele", "reference", "gain_precision", "p_brute"]].copy()
    affichage["p_brute"] = affichage["p_brute"].map(
        lambda p: "< 0,0001 %" if p == 0 else f"{100*p:.4g} %"
    )
    affichage["gain_precision"] = affichage["gain_precision"].map(lambda g: f"{100*g:+.2f} points")
    print(affichage.to_string(index=False))
    return tableau


if __name__ == "__main__":
    resultats = analyser_validation()
