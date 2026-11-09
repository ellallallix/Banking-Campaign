# %% Imports
from pathlib import Path
import matplotlib.pyplot as plt
from Statistiques import analyser_validation


def pourcentage_p(p):
    return f"{100*p:.4g} %"


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
        lignes.append([nom, f"{100*initial.gain_precision:+.2f}", pourcentage_p(initial.p_holm),
                       f"{100*simple.gain_precision:+.2f}", pourcentage_p(simple.p_holm)])
    versions = tableau.loc[tableau.version == "Comparaison des versions"].iloc[0]
    ligne_versions = [["Adaptée simplifiée − non simplifiée", f"{100*versions.gain_precision:+.2f}",
                       pourcentage_p(versions.p_holm),
                       f"[{100*versions.ic95_bas:.2f} ; {100*versions.ic95_haut:.2f}]"]]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1.6]})
        fig.suptitle("Gains de précision et p-values de validation", fontsize=18,
                     fontweight="bold", x=.06, ha="left")
        axes[0].set_title("Chaque modèle contre l’aléatoire", loc="left", fontweight="bold", pad=12)
        dessiner_tableau(axes[0], lignes,
                        ["Modèle", "Gain non simplifié\n(points)", "p-value\nnon simplifié (%)",
                         "Gain simplifié\n(points)", "p-value\nsimplifié (%)"],
                        [.30, .175, .175, .175, .175], [v[1] for v in noms.values()])
        axes[1].set_title(f"Comparaison des versions adaptées — mêmes {int(versions.clients)} clients",
                          loc="left", fontweight="bold", pad=12)
        dessiner_tableau(axes[1], ligne_versions,
                        ["Comparaison", "Gain (points)", "p-value (%)", "IC 95 % du gain (points)"],
                        [.46, .14, .14, .26], ["#c2410c"])
        fig.text(.06, .09, "Gain = différence de précision en points de pourcentage. p-values corrigées par Holm sur 9 comparaisons ; seuil : 5 %.",
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
        for _, r in tableau.iterrows():
            nom = ("Adaptée : simplifiée\n− non simplifiée" if r.version == "Comparaison des versions"
                   else f"{noms[r.modele][0]}\n{'Non simplifié' if r.version == 'Initiale' else 'Simplifié'} / aléatoire")
            calculs.append([nom, f"{int(r.gagnes)} / {int(r.perdus)}", f"{r.moyenne_d:+.4f}",
                            f"{r.ecart_type_d:.4f}", f"{100*r.erreur_standard_gain_precision:.3f}",
                            f"{100*r.gain_precision:+.2f}", f"[{100*r.ic95_bas:.2f} ; {100*r.ic95_haut:.2f}]",
                            pourcentage_p(r.p_brute), pourcentage_p(r.p_holm)])
        fig, ax = plt.subplots(figsize=(19, 10))
        fig.suptitle("Calculs statistiques — validation", x=.055, ha="left", fontsize=20, fontweight="bold")
        ax.set_title("H₀ : aucun gain positif face à la référence ; H₁ : gain positif. Comparaisons sur les mêmes clients et avec le même quota.",
                     loc="left", fontsize=11, pad=14)
        dessiner_tableau(ax, calculs,
                        ["Comparaison", "Oui gagnés\n/ perdus", "Moyenne\nde d", "Écart-type\nde d", "Erreur standard\ngain (points)",
                         "Gain\n(points)", "IC 95 % gain\n(points)", "p brute\n(%)", "p Holm\n(%)"],
                        [.23, .09, .085, .085, .105, .07, .125, .105, .105])
        fig.text(.055, .15, "d = +1 si seul le modèle retrouve un vrai Oui ; −1 si seule la référence le retrouve ; 0 sinon. M = nombre de vrais Oui ; k = nombre de clients sélectionnés.", fontsize=10, color="#334155")
        fig.text(.055, .115, "En proportions (×100 pour les points) : gain = (oui gagnés − oui perdus) / k = (M / k) × moyenne(d) ; erreur standard = (M / k) × écart-type(d) / √M.", fontsize=10, color="#334155")
        fig.text(.055, .08, "IC 95 % ≈ gain ± 1,96 × erreur standard. p brute : test exact de McNemar unilatéral (probabilité 1/2 par désaccord à gain nul). Holm corrige les 9 tests.", fontsize=10, color="#334155")
        fig.text(.055, .045, "Analyse exploratoire ; clients supposés indépendants. Les dépendances temporelles ne sont pas modélisées. Une absence de gain détecté ne prouve pas l’équivalence.", fontsize=10, color="#64748b")
        fig.tight_layout(rect=(.04, .20, .99, .93))
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
