#Exercice 2 : Kmeans Clustering avec sklearn
#Objectif : appliquer l'algorithme k-means pour regrouper le dataset en 3 clusters et analyser les centroids et l'attribution des points 

import numpy as np
from sklearn.cluster import KMeans

X= np.array([
    [2,2], #A
    [3,3], #B
    [10,10], #C
    [5,8], #D
    [8,9], #E
    [5,7], #F
    [1,2] #G
    
])

labels=['A','B','C','D','E','F','G']

# initial centroids : A,B,C
init_points = np.array([[2,2],[3,3],[10,10]])

# Créationn du modèle K-means
kmeans= KMeans(n_clusters=3, #nombre de clusters
               init=init_points, #Centroides initiaux
               n_init=1) #n_init=1 pour éviter les boucles
kmeans.fit(X)

#Affichage des résultats 
#Centroids finaux :
print("Centroids:")
for i, c in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i}: {c}")
#Attribution des points aux clusters
for i in range(len(labels)):
    print(labels[i],"->Cluster",kmeans.labels_[i])

