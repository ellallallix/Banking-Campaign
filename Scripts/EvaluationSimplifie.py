
# %% Evaluation de la version simplifiee, sans entrainement
from pathlib import Path
from Data import X_train, y_train, X_val, y_val
from EntrainementSimplifie import simplifier_donnees
from Evaluation import evaluer_modeles


if __name__ == "__main__":
    dossier_scripts = Path(__file__).resolve().parent
    _, y_train_simple = simplifier_donnees(X_train, y_train)
    X_val_simple, y_val_simple = simplifier_donnees(X_val, y_val)
    comparaison = evaluer_modeles(
        fichier_modeles=dossier_scripts / "modeles_entraines_simplifies.joblib",
        dossier_plots=dossier_scripts.parent / "Plots" / "Simplifie",
        X_val=X_val_simple, y_train=y_train_simple, y_val=y_val_simple,
    )



