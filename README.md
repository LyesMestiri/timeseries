# timeseries
Time series tools 

## Dépendances

Pré-requis : Python 3.7 ou version ultérieure.

Pour installer les dépendances

    pip install -r requirements.txt

## Exécution

Pour lancer le programme depuis le dossier 'app/' exécuter la commande :

    python main.py

## Docker

Pré-requis docker

Pour compiler l'image docker depuis le dossier 'timeseries/' exécuter la commande :

    docker build -t timeseries .

Pour lancer l'image docker exécuter la commande :

    docker run timeseries

## Structure
Le code est contenu dans le dossier '/app' :

|Fichiers|Description|
|--------|-----------|
|main.py|Test les modèles|
|models.py|Outils de distances pour les séries temporelles|
|compare.py|Fonctions permettant de comparer les résultats et temps d'exécutions|
|utils.py|Diverses fonctions utilitaires| 



## Interface

Pour lancer le logiciel, exécuter la commande

    python main.py

