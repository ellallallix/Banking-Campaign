# %% Imports — evaluation finale du modele retenu uniquement
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from Data import X_test, y_test
from Evaluation import predictions_top20
from Statistiques import comparer_aleatoire
from Random import evaluer_aleatoire


# %% Test sans recherche de parametres et sans nouvel entrainement
def tester_modele(
    fichier_modeles="modeles_entraines.joblib",
    dossier_plots="../Plots/Test",
):
    sauvegarde = load(fichier_modeles)
    informations = sauvegarde["logistique_adaptee"]
    modele = informations["modele"]
    X, y = X_test.copy(), np.asarray(y_test)
    predictions = predictions_top20(modele, X)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    aleatoire = evaluer_aleatoire(y)
    gain = comparer_aleatoire(y, predictions, aleatoire)
    texte_p = "< 0,0001 %" if gain["p_brute"] == 0 else f"{100*gain['p_brute']:.4g} %"
    parametres = informations.get("parametres_cv") or informations.get("parametres")
    resultats = {
        "modele": "Regression logistique adaptee",
        "donnees": "Initiales, non simplifiees : aucun filtrage supplementaire",
        "parametres": parametres,
        "clients_test": len(y),
        "clients_selectionnes": int(predictions.sum()),
        "vrais_oui": int(y.sum()),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "rappel": float(recall_score(y, predictions, zero_division=0)),
        "matrice_confusion": cm.tolist(),
        "comparaison_aleatoire": {
            (cle + "_pourcentage" if cle.startswith("p_") else cle):
            (100 * valeur if cle.startswith("p_") and valeur is not None else valeur)
            for cle, valeur in gain.items()
        },
    }
    dossier = Path(dossier_plots)
    dossier.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "white"}):
        fig, (ax, ax_mesures) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Test final — Régression logistique adaptée", fontsize=19, fontweight="bold", x=.06, ha="left")
        affichage = ConfusionMatrixDisplay(cm, display_labels=["Non", "Oui"])
        affichage.plot(ax=ax, cmap="Oranges", values_format="d", colorbar=True,
                       text_kw={"fontsize": 15, "fontweight": "bold"})
        ax.set_title("Données non simplifiées\nSélection de 20 % des clients", pad=18, fontweight="bold")
        ax.set_xlabel("Client sélectionné", labelpad=12)
        ax.set_ylabel("Souscription réelle", labelpad=12)
        fig.axes[-1].set_ylabel("Nombre de clients", labelpad=12)
        ax.tick_params(length=0)
        ax_mesures.axis("off")
        ax_mesures.set_title("Évaluations sur le test", loc="left", pad=18, fontweight="bold")
        mesures = [
            ["Clients évalués", str(len(y))],
            ["Clients sélectionnés", str(int(predictions.sum()))],
            ["Vrais Oui", str(int(y.sum()))],
            ["Oui retrouvés", str(gain["oui_retrouves"])],
            ["Précision@20 %", f"{resultats['precision']:.2%}"],
            ["Rappel", f"{resultats['rappel']:.2%}"],
            ["Aléatoire : précision moy. ± écart-type",
             f"{gain['precision_reference']:.2%} ± {gain['precision_reference_ecart_type']:.2%}"],
            ["Aléatoire : rappel moy. ± écart-type",
             f"{gain['rappel_reference']:.2%} ± {gain['rappel_reference_ecart_type']:.2%}"],
            ["Gain de précision", f"{100*gain['gain_precision']:+.2f} points"],
            ["SE du gain (100 tirages)", f"{100*gain['erreur_standard_gain_precision']:.3f} point"],
            ["p-value normale unilatérale", texte_p],
        ]
        grille = ax_mesures.table(cellText=mesures, colLabels=["Mesure", "Valeur"],
                                  cellLoc="left", colWidths=[.64, .36], bbox=[0, .02, 1, .96])
        grille.auto_set_font_size(False)
        grille.set_fontsize(10.5)
        for (i, j), cellule in grille.get_celld().items():
            cellule.set_edgecolor("white")
            cellule.set_linewidth(2)
            cellule.set_facecolor("#0f172a" if i == 0 else ("#fff7ed" if i % 2 else "#f8fafc"))
            if i == 0:
                cellule.set_text_props(color="white", fontweight="bold")
            elif j == 1:
                cellule.set_text_props(color="#c2410c", fontweight="bold")
        texte_parametres = "; ".join(f"{cle.removeprefix('model__')} = {valeur}" for cle, valeur in (parametres or {}).items())
        fig.text(.06, .10, f"Paramètres : {texte_parametres or 'non renseignés'}", fontsize=11, color="#334155")
        fig.text(.06, .06, "Aléatoire : 100 tirages, graines 0 à 99, chacun sélectionne 20 % des clients. Gain = précision du modèle − précision moyenne aléatoire.", fontsize=10, color="#64748b")
        fig.text(.06, .025, "SE = écart-type des gains / √100 ; p-value normale avec gain moyen / SE. Cette SE décrit uniquement la variation entre tirages.", fontsize=10, color="#64748b")
        # Reserver le bas de la figure aux parametres et aux explications.
        fig.tight_layout(rect=(.02, .24, .99, .90), w_pad=3)
        fig.savefig(dossier / "evaluation_test.png", dpi=300, bbox_inches="tight")

    print("\nTEST FINAL — LOGISTIQUE ADAPTEE, DONNEES NON SIMPLIFIEES")
    print(f"Parametres : {parametres}")
    print(f"Clients de test : {len(y)} ; selectionnes : {int(predictions.sum())}")
    print(f"Precision@20 % : {resultats['precision']:.2%}")
    print(f"Rappel : {resultats['rappel']:.2%}")
    print("Matrice de confusion :")
    print(pd.DataFrame(cm, index=["Reel Non", "Reel Oui"],
                       columns=["Non selectionne", "Selectionne"]))
    print(f"Gain de precision face a l'aleatoire : {100*gain['gain_precision']:+.2f} points")
    print(f"p-value unilaterale face a l'aleatoire : {texte_p}")
    print(f"Sorties : {dossier.resolve()}")
    plt.show()
    plt.close(fig)
    return resultats


if __name__ == "__main__":
    resultats_test = tester_modele()
