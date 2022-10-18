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