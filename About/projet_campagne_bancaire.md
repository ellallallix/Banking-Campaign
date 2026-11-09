# Projet — Ciblage d'une campagne bancaire

## Mission

Une banque souhaite sélectionner les dossiers à contacter pour proposer un dépôt à terme : un placement bancaire pour une durée convenue.

Dans notre scénario pédagogique, elle peut contacter **20 % des dossiers candidats**. Par exemple : 1 000 dossiers parmi 5 000. Cette contrainte est inventée pour le projet.

**Peut-on retrouver davantage de souscripteurs avec une sélection par modèle qu'avec une sélection aléatoire de même taille, et quelle confiance accorder au gain estimé ?**

La décision intervient **juste avant un appel dans une campagne en cours**. Toute information utilisée doit être disponible à cet instant. Tu dois examiner cette disponibilité et justifier tes choix.

## Données

- [Source officielle : UCI — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- [Télécharger l'archive officielle](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).
- Dans cette archive, ouvrir `bank-additional.zip` et retenir **`bank-additional-full.csv`**, avec sa documentation **`bank-additional-names.txt`**.
- Cette version contient **41 188 observations et 20 variables d'entrée**, classées par date de mai 2008 à novembre 2010, selon UCI.
- La **cible**, le résultat à prédire, est `y` : souscription observée, `yes` ou `no`.
- Ne pas mélanger cette version avec `bank-full.csv` ou la version réduite `bank-additional.csv`.

La fiche UCI présente plusieurs versions et mélange certaines descriptions. Pour comprendre les colonnes retenues, utiliser la documentation `bank-additional-names.txt` et signaler les ambiguïtés restantes.

Référence à citer : Moro, S., Rita, P. et Cortez, P. (2014), *Bank Marketing*, UCI Machine Learning Repository, [DOI : 10.24432/C5K306](https://doi.org/10.24432/C5K306). Licence CC BY 4.0.

## Livrables attendus — sans recette imposée

| Livrable | Ce qu'il doit établir |
|---|---|
| Cadrage | La signification d'une ligne, la cible, le moment de décision et l'utilisation envisagée. |
| Analyse des données | Qualité, distributions, relations importantes et limites de la population observée. |
| Protocole d'évaluation | Quelles observations servent à apprendre, à choisir les variantes et à évaluer le choix final ; pourquoi cette séparation convient à la mission. |
| Préparation des données | Transformations justifiées pour rendre les données utilisables par les modèles. |
| Comparaison | Au moins deux modèles que tu as étudiés et une référence simple, comparés sur les mêmes observations. |
| Analyse des résultats | Erreurs importantes, situations où la méthode fonctionne moins bien et incertitude sur le gain. |
| Recommandation | Choix argumenté, intérêt pratique, limites et vérifications nécessaires avant une utilisation réelle. |

Tu choisis les graphiques, les transformations, les modèles et les méthodes statistiques. Aucun score minimal n'est imposé. Un résultat modeste correctement évalué constitue un résultat valable.

Chaque graphique doit répondre à une question et présenter des axes, unités et légendes compréhensibles.

## Ce que la décision finale doit rendre lisible

L'évaluation principale concerne les **20 % de dossiers sélectionnés**, et pas seulement la proportion de classes correctement prédites dans tout le fichier :

- nombre de dossiers sélectionnés ;
- nombre de souscripteurs observés parmi ces dossiers ;
- proportion de tous les souscripteurs de l'ensemble évalué retrouvée par cette sélection ;
- comparaison avec une sélection aléatoire de même taille ;
- incertitude sur l'amélioration estimée, avec les hypothèses de la méthode employée.

Les références simples ne comptent pas parmi les deux modèles. Les règles de sélection et de départage des éventuelles égalités doivent être fixées avant l'évaluation finale.

## Exigences de crédibilité

- Réserver le test final avant les analyses qui orientent les choix. La documentation peut être lue au préalable. Justifier la séparation en tenant compte de l'ordre temporel annoncé et des informations réellement disponibles.
- Une **fuite d'information** apparaît lorsqu'un modèle ou un choix bénéficie d'informations indisponibles au moment de la prédiction, ou provenant indûment des données réservées. Examiner les variables, les transformations et le protocole sous cet angle.
- Les transformations qui apprennent à partir des observations doivent être ajustées sur l'entraînement de la comparaison concernée, puis appliquées telles quelles aux observations réservées.
- La validation sert à choisir les variantes ; le test final évalue le choix arrêté. Si le test inspire une modification, le signaler : il ne constitue plus une confirmation indépendante de cette nouvelle version.
- Respecter l'association entre les prédictions de modèles concernant les mêmes dossiers pour comparer leurs performances. Examiner aussi les dépendances possibles entre observations avant de quantifier l'incertitude.
- Distinguer les souscriptions observées dans ces données historiques des souscriptions qu'une nouvelle campagne provoquerait. Le fichier ne mesure pas, à lui seul, l'effet d'appeler plutôt que de ne pas appeler.
- Ne pas annoncer de bénéfice financier sans hypothèses explicites sur les coûts et revenus, ni présenter ces données anciennes comme représentatives d'une banque actuelle sans vérification.

## Présentation finale

Le rendu doit permettre à un lecteur de comprendre le raisonnement et de reproduire le travail :

- `README.md` : problème, source, démarche, résultats principaux, limites et instructions de reproduction ;
- un ou plusieurs notebooks lisibles, avec interprétations ;
- dépendances nécessaires à l'exécution ;
- figures et tableaux utiles à la conclusion.

Des scripts réutilisables peuvent compléter les notebooks si cela améliore la clarté. Les bibliothèques Python sont autorisées. Une application web, une API et une réécriture manuelle des algorithmes ne sont pas requises.

Sur le CV : **projet personnel de data science**, avec les méthodes effectivement utilisées et les résultats réellement obtenus.

## Premier rendu — Une note de cadrage d'une page maximum

Après lecture de la documentation, sans entraîner de modèle :

1. Reformule la mission et explique ce que représente une ligne du fichier.
2. Précise la cible et le moment où la décision doit être prise.
3. Identifie les informations dont la disponibilité à cet instant mérite vérification et explique tes réserves.
4. Propose une manière de mesurer honnêtement l'utilité de la sélection, notamment de réserver le test final.
5. Formule trois questions d'analyse dont les réponses pourraient changer une décision dans le projet.

Tu peux signaler une incertitude et proposer comment la résoudre. Aucun corrigé n'est fourni à l'avance. Après ce premier rendu, tu choisis la suite de la démarche.

## Accompagnement

Tu réalises le code, les analyses et les décisions. Mon rôle : relire tes productions, relever les conclusions fragiles et t'aider à identifier les notions manquantes, sans fournir spontanément de solution ou de code complet.

En cas de blocage : indice sur le concept, puis indice sur la première étape ; solution seulement si tu la demandes. Le projet mobilise les notions pertinentes, sans imposer artificiellement toutes les formules ou tous les modèles des cours.
