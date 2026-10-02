#Exercice 2 :Impact de la taille du Training Set sur le Decision Tree
#Objectif :
#Tester différentes tailles de training data(10%, 25%, 33%, 50%, 66%, 75%)
#Utiliser la méthode Random Subsampling
#Répéter 10 fois chaque expérience
#Calculer l'Average Accuracy
#Observer la relation entre: taille du training set et performance du classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#Fonction : analyser_dataset_training_size
#Elle charge le dataset,encode les attributs, teste différentes tailles de training set,applique Random Subsampling (10 répétitions)et retourne l'Average Accuracy pour chaque taille
def analyser_dataset_training_size(nom_fichier):

    #Chargement du dataset
    data = pd.read_csv(nom_fichier, header=None)

    #Séparation :
    #X = attributs descriptifs
    #Y = target attribute

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # One hot encoding uniquement sur les features
    X = pd.get_dummies(X)

    # Conversion en numpy après encoding
    X = X.values
    Y = Y.values

    #Tailles de training set à tester
    tailles =[0.10, 0.25, 0.33, 0.50, 0.66, 0.75]
    accuracies_moyennes = []
    #Test de chaque taille
    for taille in tailles:
        somme_accuracy = 0
        #Random Subsampling (10 répétitions)
        for i in range(10):
            #Séparation aléatoire
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y,
                train_size=taille
            )
            #Création du Decision Tree classifier
            clf = DecisionTreeClassifier(criterion='entropy')
            #Training
            clf.fit(X_train, y_train)
            #Test
            y_pred = clf.predict(X_test)
            #Calcul de l’Accuracy
            somme_accuracy += accuracy_score(y_test, y_pred)
        #Average Accuracy sur les 10 répétitions
        accuracy_moyenne = somme_accuracy / 10
        accuracies_moyennes.append(accuracy_moyenne)

    return tailles, accuracies_moyennes


#Liste des datasets
datasets = [("DataSets/car.data", "Car Dataset"),("DataSets/tic-tac-toe.data", "Tic-Tac-Toe Dataset"),("DataSets/zoo.data", "Zoo Dataset"),("DataSets/flag.data", "Flag Dataset"),("DataSets/agaricus-lepiota.data", "Agaricus Dataset")]


#Représentation graphique
#On représente l'Average Accuracy en fonction de la taille du training set
plt.figure(figsize=(12,8))
for i, (fichier, nom) in enumerate(datasets):
    tailles, accuracies = analyser_dataset_training_size(fichier)
    plt.subplot(2,3,i+1)
    plt.plot([t*100 for t in tailles], accuracies)
    plt.xlabel("Training size (%)")
    plt.ylabel("Average Accuracy")
    plt.title(nom)
plt.tight_layout()
plt.show()


#OBSERVATIONS : 
# OBSERVATION 1 - Car Dataset
#L'average Accuracy augemente fortement entre 10% et 25%, puis continue d'raugemtner progressovemtn jsuqu'a 75%
#La performance devient très élevée (environ 0.97) à partir de 66%
#Cela montre que plus le straining set est grand, plus le classifier apprend correctement les combinaisons d'attributs du dataset

# OBSERVATION 2 -Tic-Tac-Toe Dataset
#L'accuracy augemnte globalement avec la taillle du training set
#augemnte régulierement entre 10% et 50% puis une petite baisse à 66% suivie d'une nouvelle augmentation à 75%
#La tendance gloable reste clairement croissante
#Cela montre que plus le straining set est grand, plus le classifier apprend correctement les combinaisons d'attributs du dataset

# OBSERVATION 3 -Zoo Dataset
#L’Accuracy augmente fortement entre 10% et 25%, l'accuracy continue d'augmeenter jusqu'à 75%
#La progression devient plus fiable après 50% mais reste positive
#Cela montre que une quantité modérée de données permet déja d'obtenir une bonne performance


# OBSERVATION 4 -Flag Dataset
#l'Accuracy est faible au départ (environ 0.43)
#Elle augmente nettment entre 25% et 66%
#On observe une baisse importante à 75%
#Performance reste globaleme,t modérée(environ 0.50-0.54)
#Ce dataset semble difficile à apprendre(nombre limité d'exemples et la complexité des attribut)

# OBSERVATION 5 -Agaricus Dataset
#Contrairement aux autres datasets, l’Accuracy diminue lorsque la taille du training set augmente(La meilleure performance est obtenue avec 10%)
#Les résultats montrent une instabilité inhabituelle pour ce dataset


#OBSERVATION GÉNÉRALE
#On observe globalement une relation positive entre la taille du training set et l’Average Accuracypour la majorité des datasets(Car, Tic-Tac-Toe, Zoo)
#Plus le Decision Tree dispose d’exemples pour l’apprentissage,plus il peut construire des partitions pertinentes,ce qui améliore la capacité de généralisation

#Cependant, le cas du Agaricus et Flag Datasets montre que cette relation n’est pas toujours strictement croissante !
#Les performances peuvent varier selon la structure des données et le procédé de Random Subsampling

#Il existe donc une relation entre : la quantité de données d’entraînement et la performance du Decision Tree,mais cette relation dépend fortement des caractéristiques du dataset.
