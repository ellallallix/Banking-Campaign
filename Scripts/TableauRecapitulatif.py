# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
from Statistiques import analyser_validation


def pourcentage_p(p):
    return "< 0,0001 %" if p == 0 else f"{100*p:.4g} %"


def dessiner_tableau(ax, valeurs, entetes, largeurs, couleurs=None):
    ax.axis("off")
    grille = ax.table(cellText=valeurs, colLabels=entetes, cellLoc="center",
                      colWidths=largeurs, bbox=[0, .02, 1, .88])
    grille.auto_set_font_size(False)
    grille.set_fontsize(10)
    for (i, j), cellule in grille.get_celld().items():
        cellule.set_edgecolor("white")
        cellule.set_linewidth(2)
        cellule.set_facecolor("#0f172a" if i == 0 else ("#f1f5f9" if i % 2 else "#f8fafc"))
        if i == 0:
            cellule.set_text_props(color="white", fontweight="bold")
        elif j == 0:
            cellule.set_text_props(ha="left", fontweight="bold",
                                   color=couleurs[i-1] if couleurs else "#334155")
            cellule.PAD = .05


# %% Presentation des calculs : uniquement deux images PNG
def enregistrer_tableau(tableau, dossier_resultats="../Resultats/Statistiques"):
    dossier = Path(dossier_resultats)
    dossier.mkdir(parents=True, exist_ok=True)
    noms = {
        "SVM RBF": ("SVM RBF", "#2563eb"),
        "SVM polynomial": ("SVM polynomial", "#7c3aed"),
        "Logistique classique": ("Logistique classique", "#0f766e"),
        "Logistique adaptee": ("Logistique adaptée", "#c2410c"),
    }
    lignes = []
    for cle, (nom, _) in noms.items():
        initial = tableau.loc[(tableau.version == "Initiale") & (tableau.modele == cle)].iloc[0]
        simple = tableau.loc[(tableau.version == "Simplifiee") & (tableau.modele == cle)].iloc[0]
        lignes.append([nom, f"{100*initial.gain_precision:+.2f}", pourcentage_p(initial.p_brute),
                       f"{100*simple.gain_precision:+.2f}", pourcentage_p(simple.p_brute)])
    versions = tableau.loc[tableau.version == "Comparaison des versions"].iloc[0]
    ligne_versions = [["Adaptée simplifiée − non simplifiée", f"{100*versions.gain_precision:+.2f}",
                       pourcentage_p(versions.p_brute),
                       f"[{100*versions.ic95_bas:.2f} ; {100*versions.ic95_haut:.2f}]"]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1.6]})
        fig.suptitle("Gains de précision et p-values de validation", fontsize=18,
                     fontweight="bold", x=.06, ha="left")
        axes[0].set_title("Chaque modèle contre la moyenne de 100 tirages aléatoires", loc="left", fontweight="bold", pad=12)
        dessiner_tableau(axes[0], lignes,
                        ["Modèle", "Gain non simplifié\n(points)", "p-value\nnon simplifié (%)",
                         "Gain simplifié\n(points)", "p-value\nsimplifié (%)"],
                        [.30, .175, .175, .175, .175], [v[1] for v in noms.values()])
        axes[1].set_title(f"Comparaison des versions adaptées — mêmes {int(versions.clients)} clients",
                          loc="left", fontweight="bold", pad=12)
        dessiner_tableau(axes[1], ligne_versions,
                        ["Comparaison", "Gain (points)", "p-value (%)", "IC 95 % du gain (points)"],
                        [.46, .14, .14, .26], ["#c2410c"])
        fig.text(.06, .09, "Gain = précision du modèle − précision moyenne aléatoire (100 graines, 0 à 99). p-values brutes, sans correction.",
                 color="#475569", fontsize=10)
        fig.text(.06, .055, "Choix final : régression logistique adaptée, données non simplifiées.",
                 color="#c2410c", fontweight="bold", fontsize=11)
        fig.text(.06, .025, "Versions proches ; la simplification ne présente pas de gain détecté. Ce choix ne prouve pas la supériorité de la version initiale.",
                 color="#475569", fontsize=10)
        fig.tight_layout(rect=(.04, .13, .99, .93), h_pad=2)
        chemin_recap = dossier / "Tableau_recapitulatif.png"
        fig.savefig(chemin_recap, dpi=300, bbox_inches="tight")
        plt.close(fig)

        calculs = []
        for _, r in tableau.loc[tableau.version != "Comparaison des versions"].iterrows():
            nom = f"{noms[r.modele][0]}\n{'Non simplifié' if r.version == 'Initiale' else 'Simplifié'}"
            calculs.append([
                nom, f"{100*r.precision_reference:.2f} ± {100*r.precision_reference_ecart_type:.2f}",
                f"{100*r.gain_precision:+.2f}", f"{100*r.ecart_type_gain_precision:.3f}",
                f"{100*r.erreur_standard_gain_precision:.3f}",
                f"[{100*r.ic95_bas:.2f} ; {100*r.ic95_haut:.2f}]", pourcentage_p(r.p_brute),
            ])
        fig, axes = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={"height_ratios": [4, 1.5]})
        fig.suptitle("Calculs statistiques — validation", x=.055, ha="left", fontsize=20, fontweight="bold")
        axes[0].set_title("Face à l’aléatoire : 100 gains, un par graine ; moyenne, écart-type s et SE = s / √100",
                          loc="left", fontsize=11, pad=14)
        dessiner_tableau(axes[0], calculs,
                        ["Modèle / version", "Précision aléatoire\nmoyenne ± écart-type (%)", "Gain moyen\n(points)",
                         "Écart-type s\ndes gains (points)", "SE du gain\n(points)", "IC 95 % gain\n(points)", "p normale brute\n(%)"],
                        [.24, .20, .10, .12, .10, .14, .10])
        axes[1].set_title("Comparaison des versions adaptées : calcul apparié inchangé, sur les mêmes clients",
                          loc="left", fontsize=11, pad=14)
        r = versions
        dessiner_tableau(axes[1], [["Simplifiée − non simplifiée", f"{int(r.gagnes)} / {int(r.perdus)}",
                                   f"{r.moyenne_d:+.4f}", f"{r.ecart_type_d:.4f}",
                                   f"{100*r.erreur_standard_gain_precision:.3f}",
                                   f"{100*r.gain_precision:+.2f}", pourcentage_p(r.p_brute)]],
                        ["Comparaison", "Oui gagnés\n/ perdus", "Moyenne d", "Écart-type d", "SE (points)",
                         "Gain (points)", "p exacte brute (%)"], [.28, .14, .11, .11, .12, .10, .14])
        fig.text(.055, .155, "Aléatoire : gᵢ = précision du modèle − précision du tirage i ; gain = moyenne(gᵢ) ; s = écart-type(gᵢ), ddof=1 ; SE = s / √100.", fontsize=10, color="#334155")
        fig.text(.055, .12, "p normale = P(Z ≥ gain / SE). IC 95 % ≈ gain ± 1,96 × SE. Ces calculs décrivent la variation entre tirages, conditionnellement au modèle et aux clients fixés.", fontsize=10, color="#334155")
        fig.text(.055, .085, "Versions : d = +1 / −1 / 0 selon la capture de chaque vrai Oui ; gain = (M/k) × moyenne(d) ; SE = (M/k) × écart-type(d) / √M. p : McNemar unilatéral.", fontsize=10, color="#334155")
        fig.text(.055, .05, "M = nombre de vrais Oui ; k = nombre de clients sélectionnés. Analyse exploratoire ; une absence de gain détecté ne prouve pas l’équivalence.", fontsize=10, color="#64748b")
        fig.tight_layout(rect=(.04, .20, .99, .93), h_pad=2)
        chemin_calculs = dossier / "Calculs_statistiques.png"
        fig.savefig(chemin_calculs, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Images enregistrées : {chemin_recap.resolve()} et {chemin_calculs.resolve()}")
    return chemin_recap, chemin_calculs


# %% Recalcul depuis les donnees et les modeles sauvegardes
def generer_tableau(
    fichier_initial="modeles_entraines.joblib",
    fichier_simplifie="modeles_entraines_simplifies.joblib",
    dossier_resultats="../Resultats/Statistiques",
):
    tableau = analyser_validation(fichier_initial, fichier_simplifie)
    return enregistrer_tableau(tableau, dossier_resultats)


if __name__ == "__main__":
    fichiers = generer_tableau()
