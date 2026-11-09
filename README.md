# Ciblage d'une campagne bancaire

**Projet personnel de statistiques et de machine learning.** Comparaison sur la validation terminée ; évaluation finale sur le test préparée, non exécutée à cette étape.

Je cherche à sélectionner les **20 % de clients à contacter en priorité**, afin de retrouver le plus de souscripteurs possible dans cette sélection. Je retiens une **régression logistique adaptée, avec la version initiale non simplifiée des données**, pour le test final.

## Problème et données

La cible `y` indique la souscription d'un dépôt à terme : `yes` ou `no`. Le quota de 20 % est une contrainte choisie pour le projet. La décision intervient avant un appel, au cours d'une campagne déjà engagée : les caractéristiques doivent être disponibles à cet instant.

- **Source :** [UCI — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306).
- **Version :** `bank-additional-full.csv`, renommée `Data/data.csv` ; documentation : `Data/bank-additional-names.txt`.
- **Population :** campagnes téléphoniques d'un établissement bancaire portugais.
- **Période :** mai 2008 à novembre 2010, dans l'ordre chronologique documenté pour cette version.
- **Fichier brut :** 41 188 observations, 20 caractéristiques et une cible ; 4 640 souscriptions, soit **11,27 %**.
- **Licence des données :** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Le jeu UCI propose plusieurs fichiers : les effectifs ci-dessus concernent la version **bank-additional-full**, et non la version historique à 45 211 lignes.

## Analyse exploratoire

J'ai inspecté les types de variables, les doublons, les valeurs inconnues et les statistiques descriptives. J'ai ensuite comparé les distributions globales à celles observées parmi les souscripteurs.

Les pourcentages sont calculés séparément dans chaque population. Une proportion élevée d'une catégorie parmi les souscripteurs ne donne donc pas directement la probabilité de souscription d'un client de cette catégorie. Les associations observées ne démontrent pas une causalité.

| Variable | Observations et décisions issues de mon analyse |
|---|---|
| `age` | Distribution globalement en cloche, autour de 40 ans. J'ai envisagé une approximation normale ; les distributions globale et conditionnelle m'ont paru assez proches. |
| `duration` | Nombreux appels courts et distribution asymétrique. Les appels des souscripteurs sont souvent plus longs. Cette association est descriptive : la durée n'est pas disponible avant l'appel et est exclue de tous les modèles. |
| `pdays` | La figure limitée aux valeurs ≤ 900 présente des concentrations autour de 3 et 6 jours. Elle exclut la valeur spéciale `999`, qui signifie l'absence de contact antérieur. J'avais initialement oublié cette particularité. Pour la seconde version, je retiens l'exclusion de la colonne entière. |
| `previous` | La plupart des clients n'ont pas de contact antérieur. Je conserve cette information d'historique. |
| `campaign` | Les petits nombres de contacts dominent. J'avais supposé une faible utilité prise isolément ; je conserve la variable dans les expériences présentées. |
| `housing`, `loan`, `day_of_week` | Les distributions m'ont paru proches entre les populations. J'exclus seulement `day_of_week` dans la version simplifiée. |
| Variables économiques | Je conserve `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m` et `nr.employed`. |

La proximité visuelle de distributions ne suffit pas à établir qu'une caractéristique est inutile : les interactions, les extrémités et la dispersion peuvent compter. Les exclusions retenues sont des choix expérimentaux, pas des preuves d'absence d'information.

### Superposition de lois

J'ai superposé les proportions théoriques attendues dans les mêmes intervalles que les histogrammes :

| Variable | Population | Loi et paramètres |
|---|---|---|
| Âge | Ensemble | Normale : moyenne 40,02 ans ; écart-type 10,42 ans |
| Âge | Souscripteurs | Normale : moyenne 40,91 ans ; écart-type 13,84 ans |
| Durée | Ensemble | Exponentielle : moyenne 258,32 secondes |
| Durée | Souscripteurs | Exponentielle : moyenne 553,26 secondes |

J'ai jugé ces approximations visuellement intéressantes, avec des écarts. La loi de Poisson, essayée pour la durée, a été retirée. Aucun test d'adéquation à ces lois n'est présenté.

Figures : [âge](Plots/distribution_age.png), [durée](Plots/distribution_duration.png), [pdays](Plots/distribution_pdays.png), [previous](Plots/distribution_previous.png), [campaign](Plots/distribution_campaign.png), [catégories](Plots/distributions_categorielles.png).

## Nettoyage et deux versions des données

Les valeurs manquantes catégorielles sont représentées par la chaîne `unknown`, que `isna()` ne compte pas comme `NaN`.

| Étape commune | Nombre de lignes |
|---|---:|
| Fichier brut | 41 188 |
| Suppression de 12 lignes strictement identiques | 41 176 |
| Suppression des lignes avec `unknown` dans `job`, `marital`, `housing` ou `loan` | 39 791 |

J'ai considéré que la coïncidence de toutes les caractéristiques, notamment de la durée à la seconde près, rendait plausible une répétition d'enregistrement. Sans identifiant client, cette hypothèse ne peut pas être confirmée formellement.

Pour le premier filtre `unknown`, j'ai supposé que les lignes retirées ne représentaient pas des cas particulièrement extrêmes. La [comparaison des autres caractéristiques](Plots/comparaison_unknown.png) m'a montré des distributions globalement proches. Elle soutient visuellement mon hypothèse, sans démontrer que le retrait est sans effet.

| Version | Lignes `unknown` | Caractéristiques exclues du modèle |
|---|---|---|
| Initiale, retenue | `education` et `default` conservent `unknown` comme catégorie | `duration` |
| Simplifiée, comparaison | Toutes les lignes contenant les `unknown` restants sont retirées | `duration`, `pdays`, `day_of_week` |

La seconde version conserve les cinq caractéristiques économiques. Elle réduit aussi la population couverte : les clients exclus pour `unknown` ne reçoivent pas de prédiction dans ce protocole.

## Prétraitement et découpage

Les colonnes numériques sont standardisées avec `StandardScaler`. Les catégories sont représentées avec `OneHotEncoder(handle_unknown="ignore")`. Pour `poutcome`, je représente séparément `failure`, `success` et `nonexistent` : l'absence de campagne précédente n'est pas placée dans un ordre artificiel entre échec et succès.

| Type | Colonnes de la version initiale |
|---|---|
| Numériques | `age`, `pdays`, `previous`, `campaign`, `nr.employed`, `euribor3m`, `cons.conf.idx`, `cons.price.idx`, `emp.var.rate` |
| Catégorielles | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome` |

La version simplifiée retire `pdays` et `day_of_week` de ces sélections. Le même prétraitement est réutilisé, puis ajusté dans chaque pli de CV sur l'entraînement de ce pli. La cible est encodée séparément : `no = 0`, `yes = 1`.

Je conserve l'ordre chronologique et découpe les données nettoyées en 70 % / 15 % / 15 %. La validation et le test suivent l'entraînement dans le temps. Dans la version simplifiée, je filtre les lignes **à l'intérieur de ces périodes**, sans recalculer leurs frontières.

| Ensemble | Version initiale | Version simplifiée | Usage |
|---|---:|---:|---|
| Entraînement | 27 853 | 20 249 | Apprentissage et CV |
| Validation | 5 969 | 4 806 | Comparaison et choix |
| Test | 5 969 | Effectif calculé au lancement final | Évaluation du choix arrêté |

La CV utilise `TimeSeriesSplit(n_splits=5)` dans l'entraînement. Les proportions de souscripteurs peuvent varier entre les périodes : elles valent environ 5,62 % / 10,62 % sur l'entraînement / validation initiaux, contre 6,04 % / 11,63 % après le second filtrage. La baisse du rappel entre périodes ne suffit donc pas, isolément, à identifier un surapprentissage.

L'exploration descriptive a porté sur le fichier global. Aucun score prédictif du test n'a été utilisé pour sélectionner les modèles ou les paramètres ; ses caractéristiques et réponses ont néanmoins été incluses dans l'exploration initiale.

## Modèles et sélection des paramètres

| Modèle | Paramètres explorés |
|---|---|
| Aléatoire | Sélection uniforme de 20 % des positions sans remise |
| SVM RBF | `C` et `gamma` : `[0.01, 0.1, 1, 10]` ; `class_weight="balanced"` |
| SVM polynomial | `degree` : `[2, 3, 4]` ; `coef0` : `[10, 20]` ; `class_weight="balanced"` |
| Logistique classique, codée manuellement | `alpha` : `[0.005, 0.01, 0.1, 0.5]` ; `n_iterations` : `[500, 1000, 2500, 5000]` |
| Logistique adaptée, codée manuellement | Même grille que la classique |

Toutes les CV maximisent la **précision parmi les 20 % sélectionnés**. Les grilles sont définies dans les fichiers des modèles ; leur exécution est réservée aux scripts d'entraînement. La classe Python de chaque logistique permet son utilisation dans les pipelines de scikit-learn.

### Régression logistique adaptée

Je voulais rapprocher l'apprentissage de mon objectif de ciblage. Je calcule donc le coût et le gradient uniquement sur les 20 % de probabilités les plus élevées, en recalculant cette sélection à chaque itération.

Pour le dossier $i$, $p_i$ est la sortie sigmoïde et $y_i$ sa réponse réelle, codée 0 ou 1. Si $S_\theta$ contient les $m$ dossiers actuellement les mieux classés, le coût est :

$$
J_{20}(\theta)=-\frac{1}{m}\sum_{i\in S_\theta}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right].
$$

La version classique utilise tous les dossiers dans cette moyenne. Ce coût restreint reste un objectif intermédiaire : sa minimisation ne garantit pas directement une précision@20 % maximale. La CV compare les paramètres sur la précision@20 %, même si la méthode `score` de la classe renvoie l'opposé du coût.

Les paramètres démarrent à zéro. Les scores égaux sont départagés dans l'ordre des lignes. Limite de l'implémentation conservée : le coût adapté utilise `int(0.2*n)` alors que l'évaluation arrondit le quota au supérieur, soit une différence éventuelle d'un dossier.

## Mesures utilisées

Pour un ensemble de $n$ dossiers, je sélectionne $k=\lceil0{,}2n\rceil$ clients. Les modèles sont comparés par les scores signés les plus élevés, sans valeur absolue ni seuil fixe de 0,5.

$$
\text{Précision@20 \%}=\frac{\text{souscripteurs sélectionnés}}{k},
\qquad
\text{Rappel}=\frac{\text{souscripteurs sélectionnés}}{\text{tous les souscripteurs de l'ensemble évalué}}.
$$

À population et quota identiques, ces deux mesures donnent le même classement des modèles. Dans les matrices de confusion, « Oui » sur l'axe horizontal signifie **client sélectionné**.

## Résultats sur la validation

Les graphiques des versions initiale et simplifiée sont enregistrés respectivement dans `Plots/` et `Plots/Simplifie/`.

| Modèle | Précision initiale | Rappel initial | Précision simplifiée | Rappel simplifié |
|---|---:|---:|---:|---:|
| Aléatoire | 9,21 % | 17,35 % | 10,81 % | 18,60 % |
| SVM RBF | 14,32 % | 26,97 % | 14,76 % | 25,40 % |
| SVM polynomial | 19,68 % | 37,07 % | 20,37 % | 35,06 % |
| Régression classique | 21,44 % | 40,38 % | 17,98 % | 30,95 % |
| Régression adaptée | 22,19 % | 41,80 % | **23,18 %** | **39,89 %** |

Les colonnes initiales et simplifiées ne concernent pas exactement les mêmes clients. Elles décrivent les deux expériences ; elles ne constituent pas une comparaison statistique appariée.

### Interprétation de mes expériences

- Tous les modèles dépassent la référence aléatoire sur la validation.
- Dans la version initiale, les régressions sont proches : 265 vrais oui sélectionnés pour l'adaptée, contre 256 pour la classique, parmi 1 194 clients retenus.
- Dans la version simplifiée, l'adaptée devance davantage la classique : **223 contre 173**, parmi 962 clients retenus.
- Le RBF simplifié retrouve 100 % des souscripteurs de l'entraînement, puis 25,40 % sur la validation. Sa précision passe de 30,20 % à 14,76 %. Cet écart soutient fortement mon soupçon de surapprentissage, auquel l'évolution temporelle peut aussi contribuer.
- La simplification n'améliore pas uniformément les résultats. La classique perd notamment en précision et en rappel de validation.
- Le coût classique décroît puis se stabilise. Le coût adapté atteint une valeur basse au début, remonte, oscille et tend vers un plateau. Le renouvellement de la sélection et le pas d'apprentissage sont des explications possibles ; je n'approfondis pas ce diagnostic. Un plateau ne démontre pas la proximité d'un minimum global.

| Modèle | Paramètres retenus, initial | Paramètres retenus, simplifié |
|---|---|---|
| SVM RBF | `C=1`, `gamma=0.1` | `C=10`, `gamma=10` |
| SVM polynomial | `degree=3`, `coef0=20` | Identiques |
| Logistique adaptée | `alpha=0.5`, `n_iterations=2500` | Identiques |
| Logistique classique | `alpha=0.5`, `n_iterations=5000` | `alpha=0.005`, `n_iterations=5000` |

Les données et les variables utilisées dans les plis changent, donc le classement des hyperparamètres peut changer. Le RBF retenu devient plus local et moins contraint à rester simple. Cette observation est compatible avec le surapprentissage, sans en identifier toute la cause.

## Analyse statistique du gain et tableau récapitulatif

`TableauRecapitulatif.py` appelle `Statistiques.py`, qui charge les deux fichiers de modèles et les données de validation. Les prédictions, les gains et les p-values sont **recalculés à l'exécution**. Aucune valeur de performance n'est saisie manuellement dans le script et aucun modèle n'est réentraîné.

Le code répond à deux questions :

1. Quel gain chaque modèle apporte-t-il face au **modèle aléatoire**, dans chacune des deux versions ?
2. Quel gain la **régression adaptée simplifiée** apporte-t-elle face à la **régression adaptée non simplifiée** ?

Chaque comparaison utilise les mêmes clients et le même quota pour ses deux sélections. Pour comparer les versions, l'initiale est réévaluée sur les clients conservés par le filtre simplifié, en gardant ses colonnes d'entrée. Le gain est la différence de précision en **points de pourcentage**. Les **p-values sont affichées et exportées en pourcentage**.

Les sorties générées sont :

- [Tableau récapitulatif PNG](Resultats/Statistiques/Tableau_recapitulatif.png) : gains, p-values et choix final ;
- [Calculs statistiques PNG](Resultats/Statistiques/Calculs_statistiques.png) : désaccords entre sélections, moyenne des différences individuelles, écart-type, erreur standard, intervalles, p-values brutes/corrigées et formules utilisées.

Les deux images sont générées par `python TableauRecapitulatif.py`. Aucun rapport Markdown, CSV ou JSON n'est produit pour ces résultats.

### Méthode

Je pose $H_0$ : absence de gain positif du modèle examiné face à la référence, et $H_1$ : gain positif. Pour chaque souscripteur des mêmes clients, $d$ vaut +1 si seul le modèle le retrouve, −1 si seule la référence le retrouve, et 0 sinon.

Le test binomial exact de McNemar utilise ces désaccords pour une p-value unilatérale. Pour $M$ souscripteurs et $k$ contacts, le gain de précision vaut $(M/k)\bar d$, et son erreur standard approchée vaut $(M/k)s_d/\sqrt M$. L'image des calculs présente ces quantités et les formules utilisées.

La correction de Holm porte sur les neuf comparaisons : huit modèles contre aléatoire, puis une comparaison des versions. Le seuil est **5 %**. Les intervalles à 95 % restent marginaux et approximatifs.

Ces tests supposent des clients indépendants ; les dépendances temporelles et les clients répétés ne sont pas modélisés. La validation a déjà servi aux choix : l'analyse reste exploratoire. L'absence de gain détecté de la simplification ne démontre pas l'équivalence. Une p-value n'est pas la probabilité que $H_0$ soit vraie, et un gain statistique ne garantit pas un gain financier.

## Choix final et test

Je retiens la **régression logistique adaptée sur les données non simplifiées**, avec `alpha=0.5` et `n_iterations=2500`. Les deux versions ont des performances proches et la simplifiée est plus rapide à entraîner ; je conserve néanmoins la version initiale pour cette expérience finale, sans présenter ce choix comme une supériorité statistiquement démontrée.

`Test.py` charge le modèle adapté initial et la référence aléatoire dans `modeles_entraines.joblib`. Il utilise directement `X_test` et `y_test` issus de `Data.py`, **sans le filtrage supplémentaire de la version simplifiée**. `pdays`, `day_of_week` et les catégories `unknown` restantes sont donc conservés ; `duration` reste exclue par le prétraitement. Aucun paramètre n'est recherché et aucun modèle n'est réentraîné.

Le script enregistre une seule image, `Plots/Test/evaluation_test.png`, réunissant la matrice de confusion, les paramètres, les effectifs, la précision@20 %, le rappel et le gain/p-value face à l'aléatoire. Les p-values sont en pourcentage. Ces mesures sont également affichées dans la console ; aucun CSV ou JSON n'est généré.

**Résultats de test : non calculés à cette étape.** Le script a été vérifié sur des données simulées. Il reste à le lancer une fois sur le vrai test, à analyser ses résultats et à les ajouter au bilan, en conservant les choix arrêtés.

## Organisation et reproduction

| Fichier | Rôle |
|---|---|
| `Scripts/Data.py` | Chargement, nettoyage commun, découpage et encodage ; inspection et figures au lancement direct |
| `Scripts/Pipeline.py` | Prétraitement commun |
| `Scripts/SVM.py` | Définitions des CV des deux SVM |
| `Scripts/LogisticRegression.py` | Logistique adaptée et CV |
| `Scripts/LogisticRegressionBase.py` | Logistique classique et CV |
| `Scripts/Random.py` | Référence aléatoire |
| `Scripts/Entrainement.py`, `Scripts/EntrainementSimplifie.py` | Exécution des apprentissages et sauvegardes séparées |
| `Scripts/Evaluation.py`, `Scripts/EvaluationSimplifie.py` | Évaluation sur entraînement et validation, paramètres et figures |
| `Scripts/Statistiques.py` | Calculs de gain et p-values de validation, sans export de fichier |
| `Scripts/TableauRecapitulatif.py` | Recalcul des statistiques et génération des deux PNG de résultats et de calculs |
| `Scripts/Test.py` | Évaluation finale du choix retenu |
| `Requirements.txt` | Dépendances de l'environnement du projet |

Télécharger la version `bank-additional-full.csv` depuis UCI, la placer sous `Data/data.csv` sans réordonner les lignes, et conserver sa documentation dans `Data/`. Depuis la racine du projet sous Windows :

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
```

Les chemins relatifs sont conservés : les commandes se lancent depuis `Scripts/`. Les CV SVM peuvent prendre un temps important. Si les sauvegardes des modèles existent déjà, les scripts d'évaluation et de statistiques les chargent directement ; ils ne relancent pas l'entraînement.

Après arrêt définitif des choix, lancer l'évaluation finale :

```powershell
..\.venv\Scripts\python.exe Test.py
```

Les sauvegardes sont `Scripts/modeles_entraines.joblib` et `Scripts/modeles_entraines_simplifies.joblib`. Les figures de la seconde version sont dans `Plots/Simplifie/` et les statistiques dans `Resultats/Statistiques/`.

Le tableau, les calculs statistiques et le script de test utilisent les dépendances déjà présentes : NumPy, pandas, SciPy, Matplotlib, joblib et scikit-learn. L'accélération des SVM utilise scikit-learn-intelex. La création complète d'un environnement neuf n'a pas été revalidée à cette étape.

## Limites

- Les données concernent une seule banque et une période historique limitée. Les résultats ne garantissent pas une généralisation à une autre banque ou à une période actuelle.
- Le fichier ne fournit pas de localisation individuelle détaillée, de revenus ou de patrimoine net des clients.
- Les filtres `unknown` modifient la population couverte. Le modèle final conserve les catégories inconnues de `education` et `default`, mais pas les lignes exclues par le nettoyage commun.
- Plusieurs modifications sont appliquées ensemble dans la seconde version. Leur effet individuel n'est pas isolé.
- L'exclusion de `pdays` ne démontre pas son inutilité ; la comparaison exploratoire initiale ignorait sa valeur spéciale `999`.
- Les 20 % sont calculés sur les clients conservés. Les résultats ne décrivent pas automatiquement une campagne couvrant les clients exclus.
- Les souscriptions retrouvées ne sont pas des souscriptions causées par les appels. Aucun bénéfice monétaire n'est calculé sans hypothèses sur les coûts et la valeur des souscriptions.
- Les recherches CV, particulièrement celles des SVM, ont demandé des temps de calcul importants pendant le projet. Aucun benchmark systématique de durée ou de mémoire n'est présenté.
- Le surapprentissage du RBF simplifié illustre la sensibilité aux hyperparamètres et la différence possible entre apprentissage et généralisation.
- Le coût adapté ne garantit pas une précision@20 % maximale, et sa stabilisation ne garantit pas un minimum global.
- Les choix ont été faits sur une seule période de validation, sans confirmation sur plusieurs nouvelles périodes.
- Le test constitue encore l'étape finale à exécuter et à ajouter au bilan.

## Préparation du dépôt Git

`.gitignore` exclut l'environnement virtuel, les caches, les fichiers locaux, les CSV bruts de `Data/` et les modèles `.joblib`. Les figures et les tableaux de résultats restent versionnables. Les données se téléchargent depuis UCI et les modèles se régénèrent avec les scripts d'entraînement.

`.gitattributes` normalise les fins de ligne des fichiers texte. `Requirements.txt` contient les dépendances directes utilisées par les scripts, avec les versions de l'environnement du projet.

Après copie des fichiers, le dépôt peut être initialisé depuis la racine du projet :

```powershell
git init
git add .
git status
```

## Références techniques

- [Test binomial — SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
- [Test de McNemar — statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
- [Comparaison statistique de modèles — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
- [Paramètres du SVM RBF — scikit-learn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

## Auteur

Youness El Allali. Analyse, décisions expérimentales et implémentations manuelles des régressions ; assistance technique pour certaines parties du code, des graphiques et de la documentation.
