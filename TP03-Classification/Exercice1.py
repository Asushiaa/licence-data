#Exercice 1 : Impact de max-depth sur le ecision Tree
#Objectif : -Tester différentes valeurs de max_depth (3 à 10)
#Utiliser K-Fold cross valdiation (K+=10)
#Calculer l'average Accuracy du classifier
#Observer la relation entre : taille du dataset,profendeur du decision tree, accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#Fonction : analyser_dataset 
#Elle charge le dataset
#Encode les attributs (car le classifier utilise des valeurs numériques)
#Applique K-Fold Cross Validation
#Teste plusieurs profondeurs du Decision Tree
#Retourne l'average Accuracy pour chaque pronfedeur

def analyser_dataset(nom_fichier):
    #chargement du dataset
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
    #K-Fold Cross Validation 
    #K=10
    #A chaque itération :
    # 9 folds : traning set
    # 1 fold : Test set
    kfold= KFold(n_splits=10, shuffle=True, random_state=10)
    profondeurs=range(3,11)
    accuracies_moyennes=[]


    #Test de différentes valeurs de max_depths
    for profondeur in profondeurs:

        # Création du Decision Tree classifier ,criterion='entropy' : sélection des attributs par Information Gain
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=profondeur
        )

        somme_accuracy = 0

       
        # Application du K-Fold
        for train_index, test_index in kfold.split(X):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            #Construction du Decision Tree
            clf.fit(X_train, y_train)

            #Classification du Test set
            y_pred = clf.predict(X_test)

            #Calcul de l'Accuracy
            somme_accuracy += accuracy_score(y_test, y_pred)

        #Average Accuracy sur les 10 folds
        accuracy_moyenne = somme_accuracy / 10
        accuracies_moyennes.append(accuracy_moyenne)

    return profondeurs, accuracies_moyennes


#Liste des datasets :
datasets = [    ("DataSets/car.data", "Car Dataset"),("DataSets/tic-tac-toe.data", "Tic-Tac-Toe Dataset"),("DataSets/zoo.data", "Zoo Dataset"),("DataSets/flag.data", "Flag Dataset"),("DataSets/agaricus-lepiota.data", "Agaricus Dataset")]

#Représenation graphique 
# On représente l'average accuracy en focntion de la pronfedeur du decision tree pour chaque data set

plt.figure(figsize=(12, 8))
for i, (fichier, nom) in enumerate(datasets):
    profondeurs, accuracies = analyser_dataset(fichier)
    plt.subplot(2, 3, i+1)
    plt.plot(profondeurs, accuracies)
    plt.xlabel("max_depth")
    plt.ylabel("Average Accuracy (K=10)")
    plt.title(nom)

plt.tight_layout()
plt.show()


#OBSERVATIONS : 
# OBSERVATION 1 – Car Dataset
#L’Average Accuracy augmente globalement
#On observe une légère baisse à max_depth=4,puis amélrioation progressive jusqu'à max_depth = 10(environ 0.77 → 0.97
# Cela montre que le Car Dataset nécessite plusieurs niveaux de partition pour obtenir une bonne classification
#La structure du dataset contient plusieurs attributs(buying, maint,doors, persons, lug_boot, safety)
#Une faible profondeur ne suffit donc pas à séparer correctement toutes les classes
#On observe une amélioration continue,sans chute d’accuracy : l’augmentation de la profondeur améliore le classifier


# OBSERVATION 2 – Tic-Tac-Toe Dataset
# L'Accuracy augmente progressivement lorque la profendeur augmente
#Elle atteint un plateau atour de max_depth=7 à 10 ( environ 0.94)
#Cela indique qu'une pronfedeur élevée est nécessaire pour obtenir une bonne clasffication


# OBSERVATION 3 – Zoo Dataset
# L’Accuracy augmente fortement entre depth 3 et 5(environ 0.83 → 0.92), puis reste stable autour de 0.92-0.93
# Cela montre que quelques partitions suffisent pour classer correctement les animaux
#(augmenter la profondeur au-delà de 6 n’apporte presque aucun gain)
# Le dataset est donc relativement bien séparé avec peu de niveaux de Decision Tree


# OBSERVATION 4 – Flag Dataset
# L’Accuracy varie fortement lorsque la profondeur augmente (environ 0.59 → 0.49)
# Aucune tendance claire à l'amélrioation ,'est observée
# la performance globale reste faible (environ 0.57-0.60)
# Cela signifie que le dataset Flag est difficile à classifier avec un Decision Tree simple

# OBSERVATION 5 – Agaricus Dataset
# L’Accuracy augmente légèrement jusqu’à depth = 6(environ 0.62), puis diminue progressivement jusqu’à depth = 10 (environ 0.54)
# Cela indique que quelques partitions suffisent pour séparer les classes
# un arbre peu profond est plus adapté à ce dataset

# OBSERVATIONS GÉNÉRALES
# D’après les résultats obtenus sur les différents datasets,l’impact de max_depth dépend fortement des caractéristiques du dataset étudié
#Pour les datasets nécissant plusieurs combinaisons d'attribut(Comme Car et Tic-Tac-Toe) montrent une amélioration de l'accuracy lorsque la pronfeur augmente
#Pour les dataset ou les classes sont facilment séparables(comme zoo), une pronfdeur modérée suffit
#Pour les datasets petites ou bruitées (Flag), la performace reste instable