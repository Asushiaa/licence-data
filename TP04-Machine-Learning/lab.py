import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

# PARTIE I - KNN

def KNN(X, Y):
    """
    Implémentation du KNN avec K=1 (plus proche voisin)

    Idée 
    - pas de vrai apprentissage (lazy learning)
    - on compare chaque point avec tous les autres
    - on prend le plus proche et on copie sa classe
    On utilise ici le principe leave-one-out :
    chaque point est testé en utilisant tous les autres
    """

    # Calcul de la matrice des distances entre tous les points
    D = euclidean_distances(X, X)

    # On ignore la distance avec soi-même (sinon distance = 0)
    # → sinon chaque point serait son propre voisin
    np.fill_diagonal(D, np.inf)

    # Pour chaque point, on récupère l'indice du plus proche voisin
    nearest = np.argmin(D, axis=1)

    # La prédiction est simplement le label du voisin le plus proche
    Y_pred = Y[nearest]

    return Y_pred


def KNN_error(X, Y):
    """
    Calcul de l'erreur de classification
    erreur = nombre de mauvaises prédictions / total
    """

    Y_pred = KNN(X, Y)

    # comparaison entre prédictions et vraies classes
    return np.mean(Y_pred != Y)


def KNN_K(X, Y, K):
    """
    Extension du KNN pour K voisins

    Idée :
    - au lieu de prendre 1 voisin =>on prend K voisins
    - on fait un vote majoritaire

    Cela permet de réduire la variance
    """

    # Calcul des distances
    D = euclidean_distances(X, X)

    # On enlève auto-comparaison
    np.fill_diagonal(D, np.inf)

    # On récupère les indices des K plus proches voisins
    idx = np.argsort(D, axis=1)[:, :K]

    # On récupère leurs labels
    neighbors = Y[idx]

    # Vote majoritaire (classe la plus fréquente)
    Y_pred = mode(neighbors, axis=1).mode.flatten()

    return Y_pred


def KNN_K_error(X, Y, K):
    """
    Erreur pour K voisins
    """

    Y_pred = KNN_K(X, Y, K)
    return np.mean(Y_pred != Y)


# TEST IRIS

# Chargement du dataset Iris (dataset classique en ML)
data = load_iris()
X = data.data   # caractéristiques (features)
Y = data.target # classes (labels)

print("KNN maison :")

# Ici on évalue notre modèle avec leave-one-out
print("Erreur :", KNN_error(X, Y))



# Comparaison avec sklearn


# sklearn implémente déjà KNN de manière optimisée
model = KNeighborsClassifier(n_neighbors=1)

# apprentissage (ici juste stockage des données)
model.fit(X, Y)

# prédictions
Y_pred = model.predict(X)

print("\nKNN sklearn :")

# On compare avec notre implémentation
print("Erreur :", np.mean(Y_pred != Y))



# Influence du paramètre K


print("\nTest différents K :")

# On teste plusieurs valeurs de K pour observer l'effet
for K in [1, 3, 5, 7, 9]:
    print(f"K={K} -> erreur =", KNN_K_error(X, Y, K))


# PARTIE II - AUTRES CLASSIFIEURS


# On divise les données en :
# - 70% apprentissage
# - 30% test
# => permet d'évaluer la capacité de généralisation
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)


# SVM 

svm = SVC()
svm.fit(X_train, Y_train)
Y_pred_svm = svm.predict(X_test)


#  Naive Bayes 

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)


#  Decision Tree 

tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
Y_pred_tree = tree.predict(X_test)


# Calcul des erreurs


def error(y_true, y_pred):
    
    # Fonction générique pour calculer l'erreur
    
    return np.mean(y_true != y_pred)


print("\n PARTIE II :")

# On compare les performances des modèles
print("\nSVM error :", error(Y_test, Y_pred_svm))
print("Naive Bayes error :", error(Y_test, Y_pred_nb))
print("Decision Tree error :", error(Y_test, Y_pred_tree))


# 
# Matrices de confusion
# 

"""
La matrice de confusion permet d'analyser en détail les erreurs :
- Diagonale => bonnes prédictions
- Hors diagonale => erreurs
Permet de voir quelles classes sont confondues
"""

print("\nConfusion Matrix SVM:\n", confusion_matrix(Y_test, Y_pred_svm))
print("\nConfusion Matrix NB:\n", confusion_matrix(Y_test, Y_pred_nb))
print("\nConfusion Matrix Tree:\n", confusion_matrix(Y_test, Y_pred_tree))