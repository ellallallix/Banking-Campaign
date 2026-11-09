# Ciblage d'une campagne bancaire

**Objectif :** sélectionner les **20 % de clients à contacter en priorité**, pour retrouver le plus de souscripteurs possible.

**Résultat final :** la régression logistique adaptée, retenue sur les données non simplifiées, sélectionne **568 souscripteurs parmi 1 194 clients** sur le jeu de test : **47,57 % de précision et 24,85 % de rappel**.

**Démarche et apport personnel :** analyse et nettoyage des données, découpage chronologique **70 % / 15 % / 15 %**, implémentation manuelle de deux régressions logistiques — classique et adaptée avec un coût limité aux 20 % les mieux classés — puis comparaison avec deux SVM et une référence aléatoire estimée sur plusieurs tirages.

**Limite principale :** la proportion de souscripteurs varie fortement entre les périodes. Les résultats décrivent la période évaluée et ne garantissent pas les mêmes performances sur de futurs clients.

## Données

- **Source :** [UCI — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), fichier `bank-additional-full.csv`, placé dans `Data/data.csv`.
- **Contexte :** campagnes téléphoniques d'une banque portugaise, de mai 2008 à novembre 2010 ; les lignes suivent l'ordre chronologique.
- **Fichier brut :** 41 188 observations, 20 caractéristiques et une cible `y` : souscription d'un dépôt à terme, `yes` ou `no`.
- **Souscriptions :** 4 640, soit **11,27 %** des observations.
- **Licence :** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

La décision de sélection intervient avant l'appel : les caractéristiques utilisées doivent être disponibles à ce moment. C'est pourquoi **`duration`, la durée de l'appel, est exclue des modèles**.

## Analyse exploratoire

J'ai étudié les types de variables, les doublons, les valeurs inconnues et les statistiques descriptives, puis comparé les distributions globales à celles des souscripteurs. Dans les figures, les proportions sont calculées séparément dans chaque population : elles ne représentent pas directement la probabilité de souscription par catégorie.

| Variable | Observations et choix issus de mon analyse |
|---|---|
| `age` | Distribution autour de 40 ans ; j'ai envisagé une approximation normale. Les distributions globale et conditionnelle m'ont paru assez proches. |
| `duration` | Beaucoup d'appels courts ; les appels des souscripteurs sont souvent plus longs. Cette association reste descriptive puisque la durée est indisponible avant l'appel. |
| `pdays` | Hors valeur spéciale `999`, des concentrations apparaissent autour de 3 et 6 jours. `999` signifie l'absence de contact antérieur ; je choisis de retirer la colonne dans la version simplifiée. |
| `previous`, `campaign` | La plupart des clients n'ont pas de contact antérieur ; les petits nombres de contacts pendant la campagne dominent. Ces deux variables sont conservées. |
| `housing`, `loan`, `day_of_week` | Les distributions m'ont paru proches entre les populations ; seul `day_of_week` est retiré dans la version simplifiée. |
| Variables économiques | Je conserve `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m` et `nr.employed` dans les deux versions. |

Les lois superposées m'ont paru donner des approximations visuelles intéressantes, avec des écarts : une normale pour l'âge et une exponentielle pour la durée. Cela ne constitue pas un test d'adéquation.

![Distribution de l'âge et approximation normale](Plots/distribution_age.png)

![Distribution de la durée et approximation exponentielle](Plots/distribution_duration.png)

<details>
<summary>Paramètres des lois et autres distributions</summary>

| Variable | Ensemble | Souscripteurs |
|---|---|---|
| Âge — loi normale | Moyenne 40,02 ans ; écart-type 10,42 ans | Moyenne 40,91 ans ; écart-type 13,84 ans |
| Durée — loi exponentielle | Moyenne 258,32 secondes | Moyenne 553,26 secondes |

![Nombre de contacts pendant la campagne](Plots/distribution_campaign.png)

![Délai depuis le contact précédent, hors valeur 999](Plots/distribution_pdays.png)

![Nombre de contacts antérieurs](Plots/distribution_previous.png)

![Distributions des variables catégorielles](Plots/distributions_categorielles.png)

</details>

## Nettoyage et deux versions des données

| Étape commune | Lignes restantes |
|---|---:|
| Fichier brut | 41 188 |
| Retrait de 12 doublons stricts | 41 176 |
| Retrait des lignes avec `unknown` dans `job`, `marital`, `housing` ou `loan` | 39 791 |

La coïncidence de toutes les caractéristiques, notamment de la durée à la seconde près, m'a conduit à considérer les doublons comme des répétitions d'enregistrement. Cela reste une hypothèse, en l'absence d'identifiant client.

Pour les valeurs `unknown`, j'ai supposé que les lignes retirées ne représentaient pas des cas particulièrement extrêmes. La comparaison de leurs autres caractéristiques montre des distributions globalement proches de celles des lignes connues. Elle soutient visuellement mon hypothèse, sans démontrer que leur retrait est sans effet.

<details>
<summary>Comparaison des lignes avec et sans valeurs inconnues</summary>

![Comparaison des autres caractéristiques des lignes connues et inconnues](Plots/comparaison_unknown.png)

</details>

| Version | Traitement des `unknown` restants | Colonnes exclues du modèle |
|---|---|---|
| **Initiale, retenue** | `education` et `default` gardent `unknown` comme catégorie | `duration` |
| Simplifiée | Retrait des lignes contenant encore des `unknown` | `duration`, `pdays`, `day_of_week` |

La suppression de `pdays` est un choix expérimental : la ressemblance des distributions hors `999` ne prouve pas que la colonne apporte peu d'information. La simplification conserve toutes les variables économiques.

## Prétraitement et découpage

Les variables numériques sont standardisées avec `StandardScaler`, les catégories encodées avec `OneHotEncoder(handle_unknown="ignore")`, et la cible codée `no = 0`, `yes = 1`. Le prétraitement est ajusté sur l'entraînement de chaque pli de validation croisée (CV).

Le découpage conserve l'ordre chronologique : **70 % pour l'entraînement, 15 % pour la validation et 15 % pour le test**. La version simplifiée filtre les lignes à l'intérieur de ces mêmes périodes, sans déplacer leurs frontières.

| Ensemble | Version initiale | Version simplifiée |
|---|---:|---:|
| Entraînement | 27 853 | 20 249 |
| Validation | 5 969 | 4 806 |
| Test | 5 969 | Non retenu pour le test final |

La CV utilise `TimeSeriesSplit(n_splits=5)`. Ce protocole vise à apprendre sur le passé pour sélectionner des clients dans une période suivante. La proportion de souscripteurs évolue : environ **5,62 % à l'entraînement contre 10,62 % en validation** pour la version initiale, et **6,04 % contre 11,63 %** pour la simplifiée.

L'exploration descriptive a porté sur l'ensemble du fichier ; les scores prédictifs du test n'ont pas servi aux choix des modèles et paramètres.

## Modèles et apprentissage

| Modèle | Paramètres explorés par CV |
|---|---|
| Aléatoire | Sélection uniforme de 20 % des positions, sans remise |
| SVM RBF | `C` et `gamma` : `[0.01, 0.1, 1, 10]` ; `class_weight="balanced"` |
| SVM polynomial | `degree` : `[2, 3, 4]` ; `coef0` : `[10, 20]` ; `class_weight="balanced"` |
| Logistique classique | `alpha` : `[0.005, 0.01, 0.1, 0.5]` ; `n_iterations` : `[500, 1000, 2500, 5000]` |
| Logistique adaptée | Même grille que la classique |

Les deux régressions sont codées manuellement et intégrées aux pipelines. Toutes les CV choisissent les paramètres en maximisant la **précision parmi les 20 % sélectionnés**.

### Régression logistique adaptée

Pour rapprocher l'apprentissage de l'objectif de ciblage, je calcule le coût et le gradient uniquement sur les **20 % de probabilités les plus élevées**. Cette sélection est recalculée à chaque itération ; la régression classique utilise tous les clients.

Avec $p_i$ la probabilité prédite, $y_i$ la réponse réelle et $S_\theta$ les $m$ clients actuellement les mieux classés :

$$
J_{20}(\theta)=-\frac{1}{m}\sum_{i\in S_\theta}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right].
$$

La minimisation de ce coût ne garantit pas directement une précision@20 % maximale : la CV évalue donc les paramètres sur la précision@20 %. Les paramètres démarrent à zéro et les égalités initiales sont départagées dans l'ordre des lignes.

## Évaluation et comparaison

Pour chaque ensemble, je sélectionne les $k=\lceil0{,}2n\rceil$ meilleurs scores parmi $n$ clients. Deux mesures résument les résultats :

$$
\text{Précision@20 \%}=\frac{\text{souscripteurs sélectionnés}}{k},
\qquad
\text{Rappel}=\frac{\text{souscripteurs sélectionnés}}{\text{tous les souscripteurs de l'ensemble évalué}}.
$$

La référence aléatoire sélectionne uniformément 20 % des clients, sans remise. Ses performances sont estimées à partir de plusieurs sélections indépendantes, pour limiter l'effet d'une sélection particulièrement favorable ou défavorable. Les comparatifs présentent la **moyenne ± l'écart-type** de sa précision et de son rappel, séparément sur chaque ensemble. Les matrices de confusion montrent une sélection de référence, identifiée par la graine 42.

Dans les matrices de confusion, « Oui » sur l'axe horizontal signifie **client sélectionné**. Les figures ci-dessous présentent à la fois l'entraînement et la validation.

### Version initiale, non simplifiée

![Précision et rappel sur l'entraînement et la validation, données initiales](Plots/comparaison_precision_recall.png)

### Version simplifiée

![Précision et rappel sur l'entraînement et la validation, données simplifiées](Plots/Simplifie/comparaison_precision_recall.png)

### Mon analyse des résultats

- Les modèles appris dépassent l'aléatoire sur la validation. La régression adaptée obtient les meilleurs scores parmi les modèles comparés.
- Avec les données initiales, les deux régressions sont proches : **22,19 % contre 21,44 % de précision**, et **41,80 % contre 40,38 % de rappel**, pour l'adaptée et la classique.
- Avec les données simplifiées, l'écart est plus marqué : **23,18 % contre 17,98 % de précision**, et **39,89 % contre 30,95 % de rappel**.
- Les SVM présentent des écarts importants entre entraînement et validation. Le rappel du RBF initial passe de **73,12 % à 26,97 %** ; celui du RBF simplifié atteint **100 % à l'entraînement**, puis descend à **25,40 %** en validation. J'y vois un signe fort de surapprentissage, dans un contexte où les populations évoluent aussi dans le temps.
- Les résultats des deux versions restent globalement proches, avec une dégradation plus nette pour la logistique classique. Les clients évalués diffèrent après filtrage : la comparaison statistique entre versions doit donc porter sur leurs clients communs.

<details>
<summary>Paramètres choisis et matrices de confusion</summary>

Les paramètres retenus changent pour le SVM RBF et la logistique classique. Les données et variables utilisées par la CV ayant changé, son classement des paramètres peut aussi changer ; cela ne suffit pas à expliquer entièrement les écarts de performance.

![Paramètres retenus sur les données initiales](Plots/parametres_modeles.png)

![Paramètres retenus sur les données simplifiées](Plots/Simplifie/parametres_modeles.png)

![Matrice de confusion de la logistique adaptée, données initiales](Plots/confusion_logistique_adaptee.png)

![Matrice de confusion de la logistique adaptée, données simplifiées](Plots/Simplifie/confusion_logistique_adaptee.png)

![Matrice de confusion du SVM RBF simplifié, dont le rappel d'entraînement atteint 100 %](Plots/Simplifie/confusion_svm_rbf.png)

</details>

### Évolution des coûts

Le coût classique décroît puis se stabilise. Le coût adapté atteint une valeur basse au début, remonte puis tend vers un plateau, avec des fluctuations. Je n'approfondis pas davantage ce comportement ; un plateau ne démontre pas la proximité d'un minimum global.

![Évolution du coût de la logistique classique, données initiales](Plots/cout_logistique.png)

![Évolution du coût de la logistique adaptée, données initiales](Plots/cout_logistique_adaptee.png)

<details>
<summary>Coûts sur les données simplifiées</summary>

![Coût de la logistique classique, données simplifiées](Plots/Simplifie/cout_logistique.png)

![Coût de la logistique adaptée, données simplifiées](Plots/Simplifie/cout_logistique_adaptee.png)

</details>

## Analyse statistique du gain

Je cherche à mesurer :

1. Le gain de chaque modèle appris face à l'aléatoire, dans chacune des deux versions.
2. Le gain de la régression adaptée simplifiée face à la non simplifiée, sur les mêmes clients.

Le **gain est la différence de précision entre le modèle et sa référence**, en points de pourcentage. Face à l'aléatoire, j'utilise sa précision moyenne présentée dans l'évaluation. Pour les deux versions adaptées, la comparaison porte sur leurs clients communs, avec le même quota de sélection. Les **p-values brutes sont présentées en pourcentage**.

![Gains de précision, p-values et choix final](Resultats/Statistiques/Tableau_recapitulatif.png)

### Calculs utilisés

Je cherche à savoir si le gain moyen est positif : $H_0$ correspond à un gain nul ou négatif, et $H_1$ à un gain positif. J'utilise une **approximation normale**, avec une p-value unilatérale dans le sens d'une amélioration.

Les différences étudiées dépendent de la comparaison :

- **Face à l'aléatoire**, chaque différence vaut $g_i=P_{\mathrm{modèle}}-P_{\mathrm{référence},i}$, à partir des précisions de référence obtenues lors de l'évaluation. La précision du modèle est fixe : la moyenne des différences est donc sa précision moins la précision moyenne de la référence. La SE concerne ici l'estimation de cette référence moyenne, pour le modèle et les clients évalués.
- **Entre les deux versions adaptées**, je compare les décisions sur chaque vrai souscripteur commun. La différence $d_i$ vaut +1 si seule la simplifiée le sélectionne, −1 si seule l'initiale le sélectionne, et 0 si leurs décisions sont identiques. Avec $M$ souscripteurs et $k$ clients sélectionnés par version, je définis $g_i=(M/k)d_i$ : la moyenne de ces contributions donne le gain de précision de la simplifiée face à l'initiale.

À partir des $n$ différences, je calcule la moyenne, l'écart-type et l'erreur standard de la moyenne :

$$
\bar g=\frac{1}{n}\sum_{i=1}^{n}g_i,
\qquad
s_g=\sqrt{\frac{\sum_{i=1}^{n}(g_i-\bar g)^2}{n-1}},
\qquad
SE=\frac{s_g}{\sqrt n}.
$$

L'écart-type mesure la dispersion des différences ; la SE mesure la précision de l'estimation de leur moyenne. Pour comparer les versions, $n=M$. Le dénominateur $n-1$ correspond au calcul de l'écart-type avec `ddof=1`.

Je rapporte ensuite le gain moyen à sa SE. Sous l'hypothèse d'un gain moyen nul, la statistique est approximée par une loi normale centrée réduite :

$$
z=\frac{\bar g}{SE},
\qquad
p=1-\Phi(z),
$$

où $\Phi$ est la fonction de répartition de cette loi. La p-value est l'aire à droite de $z$ : sous $H_0$ et les hypothèses de l'approximation, elle mesure la probabilité d'obtenir une statistique au moins aussi favorable au modèle. Un gain positif élevé par rapport à sa SE donne une petite p-value.

L'intervalle à 95 % est approché par $\bar g\pm1{,}96\,SE$. Les calculs se font en proportions ; le gain, son écart-type, sa SE et les bornes de l'intervalle sont multipliés par 100 pour être présentés en points de pourcentage.

Pour les deux versions adaptées, le gain de la simplifiée est de **−0,31 point**, avec une p-value de **62,33 %** et un intervalle de **[−2,26 ; +1,63] points**. Ces résultats ne mettent pas en évidence un gain positif de la simplification.

<details>
<summary>Voir les calculs détaillés et les formules</summary>

![Calculs statistiques : moyenne, écart-type, erreur standard, intervalles et p-values](Resultats/Statistiques/Calculs_statistiques.png)

</details>

Cette analyse suppose des clients indépendants et reste exploratoire, puisque la validation a déjà servi aux choix. L'absence de gain détecté ne démontre pas l'équivalence ; la p-value n'est pas la probabilité que $H_0$ soit vraie.

## Choix final et test

Je retiens la **régression logistique adaptée sur les données non simplifiées**, avec **`alpha=0.5` et `n_iterations=2500`**. Les deux versions sont proches et la simplifiée est plus rapide à entraîner ; je conserve néanmoins l'initiale, sans affirmer sa supériorité statistique.

`Test.py` charge ce modèle déjà entraîné et l'applique au jeu de test initial, sans recherche de paramètres ni nouvel apprentissage. La figure produite rassemble la matrice de confusion, les mesures, les paramètres et la comparaison à l'aléatoire. **L'interprétation de ce résultat reste à compléter par mon analyse.**

![Évaluation finale de la logistique adaptée sur le jeu de test non simplifié](Plots/Test/evaluation_test.png)

## Reproduire le projet

Placer le fichier UCI dans `Data/data.csv`, en conservant l'ordre des lignes. Depuis la racine du projet sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Requirements.txt
cd Scripts
..\.venv\Scripts\python.exe Data.py
..\.venv\Scripts\python.exe Entrainement.py
..\.venv\Scripts\python.exe Evaluation.py
..\.venv\Scripts\python.exe EntrainementSimplifie.py
..\.venv\Scripts\python.exe EvaluationSimplifie.py
..\.venv\Scripts\python.exe TableauRecapitulatif.py
..\.venv\Scripts\python.exe Test.py
```

Les commandes se lancent depuis `Scripts/`. Les scripts d'entraînement sauvegardent `modeles_entraines.joblib` et `modeles_entraines_simplifies.joblib` ; s'ils existent déjà, les évaluations et statistiques les chargent directement. Les CV SVM peuvent prendre un temps important. Le test intervient après fixation du modèle et des choix de données.

<details>
<summary>Organisation des scripts</summary>

| Fichiers | Rôle |
|---|---|
| `Data.py`, `Pipeline.py` | Chargement, nettoyage, découpage, exploration et prétraitement |
| `SVM.py` | Définition des deux SVM et de leurs CV |
| `LogisticRegression.py`, `LogisticRegressionBase.py` | Régressions adaptée et classique, avec leurs CV |
| `Random.py` | Modèle de référence aléatoire |
| `Entrainement.py`, `EntrainementSimplifie.py` | Apprentissage et sauvegarde des deux versions |
| `Evaluation.py`, `EvaluationSimplifie.py` | Figures d'évaluation et paramètres retenus |
| `Statistiques.py`, `TableauRecapitulatif.py` | Calculs statistiques et figures dans `Resultats/Statistiques/` |
| `Test.py` | Évaluation finale du modèle retenu |

</details>

## Limites

- Les données concernent une seule banque et une période historique limitée ; elles ne garantissent pas les mêmes performances dans un autre contexte.
- Les filtres `unknown` changent la population couverte. Plusieurs modifications étant appliquées ensemble dans la version simplifiée, leur effet individuel n'est pas isolé.
- Le coût adapté ne garantit pas une précision@20 % maximale, ni un minimum global lorsqu'il se stabilise. Son quota arrondi à l'inférieur peut aussi différer d'un client de celui de l'évaluation, arrondi au supérieur.
- Les choix reposent sur une période de validation ; les tests statistiques ne modélisent pas les dépendances temporelles ou les éventuels clients répétés.
- L'entraînement, particulièrement la CV des SVM, demande des temps de calcul importants. Le RBF simplifié illustre aussi le risque de surapprentissage.
- Les gains étudiés mesurent des performances prédictives. Aucun bénéfice financier ni effet causal des appels n'est calculé.

## Auteur

Youness El Allali. Analyse, décisions expérimentales et implémentations manuelles des régressions ; assistance technique pour certaines parties du code, des graphiques et de la documentation.
