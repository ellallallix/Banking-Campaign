
#%% Import and Inspection / Youness
import pandas as pd



df = pd.read_csv("../Data/data.csv" , sep=";")

print(df.columns)

print(df.isna().sum()) # les valeurs manquantes sont notées UNKNOWN pas Nan , isna detecte rien 

print(df.describe()) # Valeurs analysé plus thourhoughly avec les graphiques

print(df.duplicated().sum()) # Que 12 doublons complets , on les supprime vu  qu'ils ne representent que 0.02% du dataset

df = df.drop_duplicates()


# %% Plots / Codex

from pathlib import Path
from textwrap import fill

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def tracer_lois_theoriques(ax, colonne, population, bornes, couleur, groupes=None):
    """Superpose les proportions attendues dans les memes classes que les donnees."""
    from scipy.stats import norm, expon

    nom_population = "Toutes" if population == "Toutes les observations" else "y = yes"
    if colonne == "age":
        moyenne, ecart_type = {
            "Toutes les observations": (40.02, 10.42),
            "y = yes": (40.91, 13.84),
        }[population]
        lois = [("Normale", norm(loc=moyenne, scale=ecart_type), "--",
                 f"μ={moyenne:g}, σ={ecart_type:g}")]
    elif colonne == "duration":
        moyenne = {"Toutes les observations": 258.32, "y = yes": 553.26}[population]
        lois = [
            ("Exponentielle", expon(loc=0, scale=moyenne), "--", f"moy.={moyenne:g} s"),
        ]
    else:
        return []

    lignes = []
    for nom_loi, loi, style, parametres in lois:
        probabilites = np.diff(loi.cdf(bornes))

        if groupes is None:
            positions = (bornes[:-1] + bornes[1:]) / 2
        else:
            probabilites = np.array([probabilites[a:b].sum() for a, b in groupes])
            positions = np.arange(len(groupes))

        ligne, = ax.plot(
            positions, 100 * probabilites, color=couleur, linestyle=style,
            marker="o", markersize=3, linewidth=2,
            label=f"{nom_loi} — {nom_population} ({parametres})",
        )
        lignes.append(ligne)
    return lignes


def comparer_lignes_unknown(donnees, colonnes_filtrees, dossier_plots="../Plots"):
    """Une figure pour comparer les lignes conservees et celles du filtre unknown."""
    from matplotlib.patches import Patch

    masque = donnees[colonnes_filtrees].eq("unknown").any(axis=1)
    groupes = [donnees.loc[~masque], donnees.loc[masque]]
    noms = ["Conservées", "À filtrer"]
    couleurs = ["#4C78A8", "#F28E2B"]
    if any(groupe.empty for groupe in groupes):
        print("Comparaison unknown : l'un des deux groupes est vide.")
        return None

    restantes = [col for col in donnees.columns if col not in colonnes_filtrees]
    numeriques = donnees[restantes].select_dtypes(include=np.number).columns.tolist()
    categorielles = [col for col in restantes if col not in numeriques]
    colonnes = numeriques + categorielles
    libelles = {
        "age": "Âge (années)",
        "duration": "Durée de l'appel (secondes)",
        "pdays": "Délai (jours ; 999 = jamais contacté)",
        "previous": "Nombre de contacts antérieurs",
        "campaign": "Nombre de contacts dans la campagne",
        "emp.var.rate": "Variation de l'emploi (%)",
        "cons.price.idx": "Indice des prix (sans unité)",
        "cons.conf.idx": "Indice de confiance (sans unité)",
        "euribor3m": "Euribor à 3 mois (%)",
        "nr.employed": "Emploi (unité non précisée dans la documentation)",
    }

    ncols = 4
    nrows = int(np.ceil(len(colonnes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(23, 4.5 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, colonnes):
        if col in numeriques:
            valeurs = [groupe[col].dropna().to_numpy() for groupe in groupes]
            boites = ax.boxplot(
                valeurs, orientation="horizontal", patch_artist=True,
                widths=0.5, showfliers=True, showmeans=True,
                medianprops={"color": "#172033", "linewidth": 1.5},
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "#172033", "markersize": 4},
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
            )
            for i, couleur in enumerate(couleurs):
                boites["boxes"][i].set_facecolor(couleur)
                boites["boxes"][i].set_alpha(0.7)
                boites["fliers"][i].set_markeredgecolor(couleur)
                boites["fliers"][i].set_markerfacecolor(couleur)
            ax.set_yticks([1, 2], noms, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(libelles.get(col, col), fontsize=9)
            ax.set_ylabel("Groupe", fontsize=9)
            bornes = " | ".join(
                f"{nom} : [{valeur.min():g} ; {valeur.max():g}]"
                for nom, valeur in zip(noms, valeurs)
            )
            ax.set_title(f"{col}\nMin / max — {bornes}", fontsize=9.5, pad=12)
        else:
            categories = donnees[col].astype("string").fillna("<NaN>").value_counts().index
            positions = np.arange(len(categories))
            for i, (groupe, couleur) in enumerate(zip(groupes, couleurs)):
                comptes = (
                    groupe[col].astype("string").fillna("<NaN>")
                    .value_counts().reindex(categories, fill_value=0)
                )
                ax.barh(
                    positions + (i - 0.5) * 0.38, 100 * comptes / len(groupe),
                    height=0.36, color=couleur,
                )
            ax.set_yticks(positions, [fill(str(c), 22) for c in categories], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Observations dans chaque groupe (%)", fontsize=9)
            ax.set_ylabel("Catégorie", fontsize=9)
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.set_xlim(0, 100)
            ax.set_title(col, fontsize=12, pad=12)

        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#cbd5e1", linestyle="--", alpha=0.65)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.flat[len(colonnes):]:
        ax.set_visible(False)

    fig.suptitle(
        "Valeurs inconnues : comparaison avant suppression des lignes\n"
        "Groupes définis par job, marital, housing et loan",
        fontsize=18, fontweight="bold", y=0.995,
    )
    fig.legend(
        handles=[Patch(color=c, label=f"{nom} (n = {len(groupe):,})")
                 for nom, groupe, c in zip(noms, groupes, couleurs)],
        loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=2,
        frameon=False, fontsize=13,
    )
    fig.text(
        0.5, 0.012,
        "Boîtes : médiane et quartiles ; losange blanc : moyenne ; points : valeurs au-delà des moustaches.\n"
        "Les 4 colonnes du filtre sont exclues de cette comparaison. La durée reste uniquement descriptive.",
        ha="center", fontsize=10, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94), h_pad=3, w_pad=2)
    dossier = Path(dossier_plots)
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / "comparaison_unknown.png"
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    print(f"Comparaison unknown enregistrée : {chemin.resolve()}")
    plt.show()
    plt.close(fig)
    return chemin


if __name__ == "__main__":

    Path("../Plots/").mkdir(parents=True, exist_ok=True)

    quantitatives = ["age", "duration", "pdays", "previous", "campaign"]

    categorielles = [
        "job", "marital", "education", "default", "housing",
        "loan", "contact", "month", "day_of_week", "poutcome", "y"
    ]

    libelles = {
        "age": "Âge (ans)",
        "duration": "Durée (secondes)",
        "pdays": "Nombre de jours",
        "previous": "Nombre de contacts antérieurs",
        "campaign": "Nombre de contacts pendant la campagne"
    }

    couleur_toutes = "#4C78A8"
    couleur_yes = "#F28E2B"

    masque_yes = df["y"].eq("yes")


    def styliser_grille(ax, axe="y"):
        ax.set_axisbelow(True)
        ax.grid(
            axis=axe,
            color="#AAB4C0",
            linestyle="--",
            linewidth=0.85,
            alpha=0.8
        )


    def pourcentages(effectifs, total):
        effectifs = np.asarray(effectifs, dtype=float)

        if total == 0:
            return np.zeros_like(effectifs)

        return 100 * effectifs / total


    def resumer(valeurs):
        if valeurs.empty:
            return "Aucune observation"

        return (
            f"Moyenne : {valeurs.mean():.2f}   |   "
            f"Médiane : {valeurs.median():.2f}   |   "
            f"Écart-type : {valeurs.std(ddof=1):.2f}"
        )


    def regrouper_classes(effectifs, total_reference=None):
        """
        Chaque groupe fusionné aux extrémités :
        - représente moins de 5 % du total ;
        - ne dépasse pas la hauteur de son voisin vers le centre.
        """
        effectifs = np.asarray(effectifs)
        n = len(effectifs)

        if n == 0:
            return []

        total = (
            effectifs.sum()
            if total_reference is None
            else total_reference
        )

        if total == 0:
            return [(i, i + 1) for i in range(n)]

        centrales = np.flatnonzero(effectifs * 100 >= 5 * total)

        if len(centrales):
            debut = int(centrales[0])
            fin = int(centrales[-1]) + 1
        else:
            debut = int(np.argmax(effectifs))
            fin = debut + 1

        def parcourir(indices, voisin):
            groupes = []
            i = 0

            while i < len(indices):
                j = i + 1
                somme = effectifs[indices[i]]

                while j < len(indices):
                    nouvelle_somme = somme + effectifs[indices[j]]

                    if (
                        nouvelle_somme * 100 >= 5 * total
                        or nouvelle_somme > voisin
                    ):
                        break

                    somme = nouvelle_somme
                    j += 1

                membres = indices[i:j]
                groupes.append((min(membres), max(membres) + 1))
                voisin = somme
                i = j

            return groupes

        gauche = parcourir(
            list(range(debut - 1, -1, -1)),
            effectifs[debut]
        )
        droite = parcourir(
            list(range(fin, n)),
            effectifs[fin - 1]
        )
        centre = [(i, i + 1) for i in range(debut, fin)]

        return gauche[::-1] + centre + droite


    # ============================================================
    # QUANTITATIVES : un fichier par caractéristique
    # ============================================================

    for col in quantitatives:
        valides = df[col].notna()

        if col == "pdays":
            valides = valides & df[col].le(900)

        toutes = df.loc[valides, col]
        oui = df.loc[valides & masque_yes, col]

        if toutes.empty:
            raise ValueError(f"Aucune valeur disponible pour {col}.")

        etiquettes = None

        if col == "age":
            bornes = np.histogram_bin_edges(toutes, bins=22)

        elif col == "pdays":
            minimum = int(toutes.min())
            maximum = int(toutes.max())
            bornes = np.arange(minimum - 0.5, maximum + 1.5, 1)

        elif col == "previous":
            etiquettes = ["0", "1", "[2, 7]"]

        elif col == "campaign":
            etiquettes = ["1", "2", "3", "4", "5", "6", "7–10", ">10"]

        else:
            debut = np.floor(toutes.min() / 100) * 100
            fin = (np.floor(toutes.max() / 100) + 1) * 100
            bornes = np.arange(debut, fin + 100, 100)

            effectifs, _ = np.histogram(toutes, bins=bornes)
            groupes = regrouper_classes(effectifs)

            etiquettes = []

            for a, b in groupes:
                fermeture = "]" if b == len(effectifs) else "["
                etiquettes.append(
                    f"[{bornes[a]:g} ; {bornes[b]:g}{fermeture}"
                )

        fig, (ax_hist, ax_boite) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            gridspec_kw={"width_ratios": [3, 1]}
        )

        populations = [
            ("Toutes les observations", toutes, couleur_toutes),
            ("y = yes", oui, couleur_yes)
        ]

        for numero, (nom, valeurs, couleur) in enumerate(
            populations, start=1
        ):
            if valeurs.empty:
                continue

            if etiquettes is None:
                ax_hist.hist(
                    valeurs,
                    bins=bornes,
                    weights=np.full(len(valeurs), 100 / len(valeurs)),
                    color=couleur,
                    alpha=0.55,
                    edgecolor="white",
                    label=nom
                )

            else:
                if col == "previous":
                    comptes = [
                        valeurs.eq(0).sum(),
                        valeurs.eq(1).sum(),
                        valeurs.between(2, 7).sum()
                    ]

                elif col == "campaign":
                    comptes = [
                        valeurs.eq(i).sum() for i in range(1, 7)
                    ] + [
                        valeurs.between(7, 10).sum(),
                        valeurs.gt(10).sum()
                    ]

                else:
                    comptes_initiaux, _ = np.histogram(
                        valeurs, bins=bornes
                    )
                    comptes = [
                        comptes_initiaux[a:b].sum()
                        for a, b in groupes
                    ]

                positions = np.arange(len(etiquettes))
                decalage = (numero - 1.5) * 0.4

                ax_hist.bar(
                    positions + decalage,
                    pourcentages(comptes, len(valeurs)),
                    width=0.38,
                    color=couleur,
                    edgecolor="white",
                    label=nom
                )

            if col in ("age", "duration"):
                tracer_lois_theoriques(
                    ax_hist, col, nom, bornes, couleur,
                    groupes=groupes if col == "duration" else None,
                )

            boite = ax_boite.boxplot(
                valeurs,
                positions=[numero],
                widths=0.55,
                whis=1.5,
                showfliers=True,
                patch_artist=True,
                manage_ticks=False,
                medianprops={"color": "#222222", "linewidth": 1.5}
            )

            boite["boxes"][0].set_facecolor(couleur)
            boite["boxes"][0].set_alpha(0.65)
            boite["fliers"][0].set_markeredgecolor(couleur)
            boite["fliers"][0].set_alpha(0.4)

        if etiquettes is None:
            if col == "pdays":
                ax_hist.set_xticks(np.arange(minimum, maximum + 1))
                ax_hist.tick_params(axis="x", labelsize=9)
        else:
            ax_hist.set_xticks(np.arange(len(etiquettes)))
            ax_hist.set_xticklabels(
                etiquettes,
                rotation=60 if col == "duration" else 0,
                ha="right" if col == "duration" else "center",
                fontsize=9 if col == "duration" else 11
            )

        ax_hist.set_xlabel(libelles[col])
        ax_hist.set_ylabel("Observations dans chaque population (%)")
        ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        if col in ("age", "duration"):
            ax_hist.legend(fontsize=8, ncol=2)
        else:
            ax_hist.legend()
        styliser_grille(ax_hist)

        ax_boite.set_xticks([1, 2])
        ax_boite.set_xticklabels(["Toutes", "y = yes"])
        ax_boite.set_xlim(0.5, 2.5)
        ax_boite.set_ylabel(libelles[col])
        styliser_grille(ax_boite)

        titre = col if col != "pdays" else "pdays — valeurs ≤ 900"

        fig.suptitle(
            f"{titre}\n"
            f"Toutes les observations — {resumer(toutes)}\n"
            f"y = yes — {resumer(oui)}",
            fontsize=12
        )

        if col in ("age", "duration"):
            fig.text(
                0.5, 0.012,
                "Traits : proportions théoriques attendues dans chaque classe ; couleurs : populations.",
                ha="center", fontsize=8,
            )
        fig.tight_layout(rect=(0, 0.04 if col in ("age", "duration") else 0, 1, 0.84))
        fig.savefig(
            f"../Plots/distribution_{col}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()
        plt.close(fig)


    # ============================================================
    # CATÉGORIELLES : uniquement la distribution globale pour y
    # ============================================================

    fig, axes = plt.subplots(4, 3, figsize=(24, 24), sharex=True)

    for ax, col in zip(axes.flat, categorielles):
        effectifs = df[col].value_counts(
            dropna=False
        ).sort_values(ascending=False)

        total = effectifs.sum()

        if total == 0:
            ax.set_visible(False)
            continue

        if col == "y":
            # Aucun ajout conditionnel pour la cible.
            proportions = pourcentages(effectifs.to_numpy(), total)
            positions = np.arange(len(effectifs))

            ax.barh(
                positions,
                proportions,
                height=0.6,
                color=couleur_toutes
            )
            ax.set_yticks(positions)
            ax.set_yticklabels(
                [str(nom) for nom in effectifs.index],
                fontsize=11
            )
            ax.set_xlabel("Observations (%)")

        else:
            total_yes = int(masque_yes.sum())

            effectifs_yes = (
                df.loc[masque_yes, col]
                .value_counts(dropna=False)
                .reindex(effectifs.index, fill_value=0)
            )

            proteges = np.array([
                col == "education"
                and str(nom).strip().lower() in ("unknown", "illiterate")
                for nom in effectifs.index
            ])

            effectifs_proteges = effectifs[proteges]
            effectifs_yes_proteges = effectifs_yes[proteges]

            effectifs_a_regrouper = effectifs[~proteges]
            effectifs_yes_a_regrouper = effectifs_yes[~proteges]

            etiquettes = [
                str(nom) for nom in effectifs_proteges.index
            ]
            comptes_toutes = effectifs_proteges.to_list()
            comptes_yes = effectifs_yes_proteges.to_list()

            groupes = regrouper_classes(
                effectifs_a_regrouper.to_numpy(),
                total_reference=total
            )

            for a, b in groupes:
                noms = [
                    str(nom)
                    for nom in effectifs_a_regrouper.index[a:b]
                ]

                etiquette = (
                    noms[0]
                    if b - a == 1
                    else "Groupe : " + ", ".join(noms)
                )

                etiquettes.append(etiquette)
                comptes_toutes.append(
                    effectifs_a_regrouper.iloc[a:b].sum()
                )
                comptes_yes.append(
                    effectifs_yes_a_regrouper.iloc[a:b].sum()
                )

            proportions_toutes = pourcentages(comptes_toutes, total)
            proportions_yes = pourcentages(comptes_yes, total_yes)

            ordre = np.argsort(-proportions_toutes, kind="stable")

            proportions_toutes = proportions_toutes[ordre]
            proportions_yes = proportions_yes[ordre]
            etiquettes = [
                fill(etiquettes[i], width=32)
                for i in ordre
            ]

            positions = np.arange(len(etiquettes))

            ax.barh(
                positions - 0.2,
                proportions_toutes,
                height=0.38,
                color=couleur_toutes,
                label="Toutes les observations"
            )
            ax.barh(
                positions + 0.2,
                proportions_yes,
                height=0.38,
                color=couleur_yes,
                label="y = yes"
            )

            ax.set_yticks(positions)
            ax.set_yticklabels(etiquettes, fontsize=11)
            ax.set_xlabel("Observations dans chaque population (%)")

        ax.invert_yaxis()
        ax.set_title(col, fontsize=14)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.tick_params(axis="x", labelbottom=True)
        styliser_grille(ax, axe="x")

    for ax in axes.flat[len(categorielles):]:
        ax.set_visible(False)

    poignees, noms_legende = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        poignees,
        noms_legende,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
        fontsize=13
    )

    fig.suptitle("Distributions catégorielles", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        "../Plots/distributions_categorielles.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)  

# %% Valeurs Manquantes / Youness

colonnes_a_filtrer = ["job", "marital", "housing", "loan"]

if __name__ == "__main__":
    comparer_lignes_unknown(df, colonnes_a_filtrer)

nb_lignes_avant = len(df)

df = df.loc[
    ~df[colonnes_a_filtrer].eq("unknown").any(axis=1)
].copy()

print(f"Nombre de lignes supprimées : {nb_lignes_avant - len(df)}")

unknown_restants = df.eq("unknown").sum()

print(unknown_restants[unknown_restants > 0])
print(f"Total de valeurs unknown restantes : {unknown_restants.sum()}")

# %% Data Split / Youness

from sklearn.preprocessing import LabelEncoder


X = df[[ col for col in df.columns if col != 'y']]
y = df["y"]

n = len(X)

# Découpage chronologique 

X_train , X_val , X_test = X[:int(0.7*n)] , X[int(0.7*n):int(0.85*n)],X[int(0.85*n):]
y_train ,y_val,y_test= y[:int(0.7*n)] , y[int(0.7*n):int(0.85*n)],y[int(0.85*n):]

# l'ordre du dataset est chronologique de mai 2008 à nov 2010, comme documenté par UCI


encodeur_y = LabelEncoder()

y_train = encodeur_y.fit_transform(y_train)
y_val = encodeur_y.transform(y_val)
y_test = encodeur_y.transform(y_test)


# %%



