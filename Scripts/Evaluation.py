
# %% Import / Youness
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from textwrap import fill
from matplotlib.ticker import MaxNLocator
from joblib import load
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
)
from Data import y_train, X_val, y_val
from Random import evaluer_aleatoire


# %% Decisions de selection, sans entrainement
def predictions_top20(modele, X, aleatoire=False):
    if aleatoire:
        return np.asarray(modele.predict(X), dtype=int)

    if hasattr(modele, "decision_function"):
        scores = np.asarray(modele.decision_function(X))
    else:
        scores = np.asarray(modele.predict_proba(X))[:, 1]

    k = max(1, int(np.ceil(0.20 * len(scores))))
    selection = np.argsort(-scores, kind="stable")[:k]
    predictions = np.zeros(len(scores), dtype=int)
    predictions[selection] = 1
    return predictions


# %% Style commun aux tableaux
def styliser_tableau(tableau, couleurs_lignes):
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(10.5)
    for (ligne, colonne), cellule in tableau.get_celld().items():
        cellule.set_edgecolor("white")
        cellule.set_linewidth(2)
        cellule.PAD = 0.08
        if ligne == 0:
            cellule.set_facecolor("#0f172a")
            cellule.set_text_props(color="white", fontweight="bold")
        else:
            cellule.set_facecolor("#f1f5f9" if ligne % 2 else "#f8fafc")
            if colonne == 0:
                cellule.set_text_props(
                    ha="left", fontweight="bold", color=couleurs_lignes[ligne - 1],
                )


# %% Evaluation : matrices de confusion et courbes de cout
def evaluer_modeles(
    fichier_modeles="modeles_entraines.joblib", dossier_plots=None,
    X_val=X_val, y_train=y_train, y_val=y_val,
):
    # ../Plots est resolu par rapport a ce script, pas au dossier de lancement.
    if dossier_plots is None:
        dossier_plots = Path(__file__).resolve().parent.parent / "Plots"
    dossier_plots = Path(dossier_plots)
    dossier_plots.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": "#334155",
        "text.color": "#0f172a",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "savefig.facecolor": "white",
    })
    resultats = load(fichier_modeles)
    references = {
        "entrainement": evaluer_aleatoire(y_train),
        "validation": evaluer_aleatoire(y_val),
    }
    lignes = []
    lignes_parametres = []
    noms = {
        "svm_rbf": "SVM RBF",
        "svm_poly": "SVM polynomial",
        "logistique_adaptee": "Régression logistique adaptée",
        "aleatoire": "Aléatoire",
        "logistique": "Régression logistique classique",
    }
    couleurs = {
        "svm_rbf": "#2563eb",
        "svm_poly": "#7c3aed",
        "logistique_adaptee": "#c2410c",
        "aleatoire": "#475569",
        "logistique": "#0f766e",
    }
    palettes = {
        "svm_rbf": "Blues",
        "svm_poly": "Purples",
        "logistique_adaptee": "Oranges",
        "aleatoire": "Greys",
        "logistique": "Greens",
    }

    for nom, informations in resultats.items():
        titre = noms.get(nom, nom)
        modele = informations["modele"]
        parametres = informations.get("parametres_cv")
        if parametres is None:
            parametres = informations.get("parametres")
        print(f"\n{titre}")
        print(f"Parametres : {parametres or 'Aucun parametre ajuste'}")
        texte_parametres = "\n".join(
            fill(f"{cle.removeprefix('model__')} = {valeur}", width=75)
            for cle, valeur in (parametres or {}).items()
        ) or "100 tirages : graines 0 à 99 ; matrice de référence : graine 42"
        lignes_parametres.append([titre, texte_parametres])

        # Les predictions d'entrainement ont deja ete conservees.
        pred_train = np.asarray(informations["predictions_train"], dtype=int)
        pred_val = predictions_top20(modele, X_val, aleatoire=(nom == "aleatoire"))

        cm_train = confusion_matrix(y_train, pred_train, labels=[0, 1])
        cm_val = confusion_matrix(y_val, pred_val, labels=[0, 1])
        maximum = max(cm_train.max(), cm_val.max(), 1)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        fig.suptitle(
            f"{titre}\nSélection de 20 % des clients"
            + (" — tirage de référence, graine 42" if nom == "aleatoire" else ""),
            fontsize=16, fontweight="bold",
        )
        affichages = []
        for ax, periode, cm in (
            (axes[0], "Entraînement", cm_train),
            (axes[1], "Validation", cm_val),
        ):
            affichage = ConfusionMatrixDisplay(cm, display_labels=["Non", "Oui"])
            affichage.plot(
                ax=ax, cmap=palettes.get(nom, "Blues"),
                values_format="d", colorbar=False,
                im_kw={"vmin": 0, "vmax": maximum},
                text_kw={"fontsize": 14, "fontweight": "bold"},
            )
            for i in range(2):
                for j in range(2):
                    affichage.text_[i, j].set_color(
                        "white" if cm[i, j] > maximum / 2 else "#0f172a"
                    )
            affichages.append(affichage)
            ax.set_title(periode, pad=14)
            ax.set_xlabel("Client sélectionné", labelpad=12)
            ax.set_ylabel("Souscription réelle", labelpad=12)
            ax.tick_params(length=0, pad=8)
            ax.grid(False)
        barre = fig.colorbar(affichages[0].im_, ax=axes, shrink=0.78, pad=0.04)
        barre.set_label("Nombre de clients", labelpad=12)
        barre.locator = MaxNLocator(integer=True)
        barre.update_ticks()
        fig.savefig(dossier_plots / f"confusion_{nom}.png", dpi=300, bbox_inches="tight")

        couts = informations.get("couts")
        if couts is not None and len(couts) > 0:
            fig_cout, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
            ax.plot(
                np.arange(1, len(couts) + 1), couts,
                color=couleurs.get(nom, "#2563eb"), linewidth=2.4,
                label=("Coût d'entraînement sur tous les clients" if nom == "logistique"
                       else "Coût d'entraînement sur les 20 % sélectionnés"),
            )
            ax.set_title(f"{titre}\nÉvolution du coût d'entraînement", loc="left", pad=18)
            ax.set_xlabel("Itération", labelpad=10)
            ax.set_ylabel("Coût logistique moyen\n(sans unité)", labelpad=12)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=10)
            fig_cout.savefig(dossier_plots / f"cout_{nom}.png", dpi=300, bbox_inches="tight")
        else:
            print("Aucun historique de cout enregistre pour ce modele.")

        ligne = {
            "Modele": titre,
            "Precision entrainement": precision_score(y_train, pred_train, zero_division=0),
            "Recall entrainement": recall_score(y_train, pred_train, zero_division=0),
            "Precision validation": precision_score(y_val, pred_val, zero_division=0),
            "Recall validation": recall_score(y_val, pred_val, zero_division=0),
        }
        for periode, reference in references.items():
            if nom == "aleatoire":
                ligne[f"Precision {periode}"] = reference["precision_moyenne"]
                ligne[f"Recall {periode}"] = reference["rappel_moyen"]
            ligne[f"Ecart-type precision {periode}"] = (
                reference["precision_ecart_type"] if nom == "aleatoire" else 0.0
            )
            ligne[f"Ecart-type rappel {periode}"] = (
                reference["rappel_ecart_type"] if nom == "aleatoire" else 0.0
            )
        ligne["Gain precision validation"] = (
            ligne["Precision validation"] - references["validation"]["precision_moyenne"]
        )
        lignes.append(ligne)

    # Comparaison des performances
    comparaison = pd.DataFrame(lignes).set_index("Modele")
    couleurs_lignes = [couleurs.get(nom, "#334155") for nom in resultats]
    print("\nCOMPARAISON - SELECTION DE 20 % DES CLIENTS")
    print(comparaison.to_string(
        formatters={colonne: "{:.2%}".format for colonne in comparaison.columns}
    ))

    # Version visuelle du meme tableau, avec des valeurs exprimees en pourcentage.
    fig_tableau, ax = plt.subplots(figsize=(17, 4.5))
    ax.axis("off")
    ax.set_title("Comparaison des modèles", loc="left", fontsize=17, pad=24)
    ax.text(
        0, 1.02, "Sélection de 20 % des clients ; aléatoire : moyenne ± écart-type sur 100 graines (0 à 99)",
        transform=ax.transAxes, color="#64748b", fontsize=11,
    )
    mesures = ["Precision entrainement", "Recall entrainement",
               "Precision validation", "Recall validation"]
    dispersions = ["Ecart-type precision entrainement", "Ecart-type rappel entrainement",
                   "Ecart-type precision validation", "Ecart-type rappel validation"]
    valeurs = []
    for nom_modele, ligne in comparaison.iterrows():
        valeurs.append(
            [nom_modele] + [
                f"{ligne[col]:.2%}" + (f" ± {ligne[std]:.2%}" if nom_modele == "Aléatoire" else "")
                for col, std in zip(mesures, dispersions)
            ] + [f"{100*ligne['Gain precision validation']:+.2f}"]
        )
    tableau = ax.table(
        cellText=valeurs,
        colLabels=["Modèle", "Précision\nEntraînement", "Rappel\nEntraînement",
                   "Précision\nValidation", "Rappel\nValidation", "Gain validation\n(points)"],
        cellLoc="center", colLoc="center",
        colWidths=[0.28, 0.15, 0.15, 0.15, 0.15, 0.12],
        bbox=[0, 0.04, 1, 0.84],
    )
    styliser_tableau(tableau, couleurs_lignes)
    ax.text(0, -.07, "Gain = précision de validation − précision moyenne de l’aléatoire.",
            transform=ax.transAxes, fontsize=10, color="#64748b")
    fig_tableau.tight_layout()
    fig_tableau.savefig(
        dossier_plots / "comparaison_precision_recall.png", dpi=300, bbox_inches="tight"
    )

    # Tableau des parametres choisis
    nb_lignes = max(texte.count("\n") + 1 for _, texte in lignes_parametres)
    fig_parametres, ax = plt.subplots(
        figsize=(13, max(3.5, 0.3 * (nb_lignes + 1) * len(lignes_parametres) + 1))
    )
    ax.axis("off")
    ax.set_title("Paramètres des modèles", loc="left", fontsize=17, pad=24)
    tableau_parametres = ax.table(
        cellText=lignes_parametres,
        colLabels=["Modèle", "Paramètres choisis par CV / fixés"],
        cellLoc="left", colLoc="left", colWidths=[0.32, 0.68],
        bbox=[0, 0.04, 1, 0.9],
    )
    styliser_tableau(tableau_parametres, couleurs_lignes)
    fig_parametres.tight_layout()
    fig_parametres.savefig(
        dossier_plots / "parametres_modeles.png", dpi=300, bbox_inches="tight"
    )
    print(f"\nFigures enregistrées dans : {dossier_plots.resolve()}")
    plt.show()
    return comparaison


# %% Execution directe
if __name__ == "__main__":
    comparaison = evaluer_modeles()



