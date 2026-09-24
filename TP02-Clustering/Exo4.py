#Exercice 4 : Analyse de l'inertia pour le dataset google_review_ratings
#Objectif : Utilsier la valeur d'inertia pour justifier le choix du nombre optimal de clusters avec k-means

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#charger le dataset 
review= pd.read_csv("google_review_ratings.csv")

#Garder uniquement les collonnes numériques
X=review.select_dtypes(include="number")

#remplacer les valeurs manquantes par la moyenne de chaque colonne
X.fillna(X.mean(),inplace=True)

#Calculer inertia pour n_clusters de 3 à 10
inertia_values = []
for n in range(3,11):
    model=KMeans(n_clusters=n,n_init=10,random_state=0)
    model.fit(X)
    inertia_values.append(model.inertia_)


#Afficher les résultats 
for n, inertia in zip(range(3,11), inertia_values):
    print(f"n_clusters = {n} -> inertia = {inertia: .2f} ")


#Yracer le graphe pour visualiser 
plt.figure(figsize=(8,5))
plt.plot(range(3,11),inertia_values,'g*-')
plt.title("Inertia vs Nombre de clusters (n_clusters)")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertia")
plt.xticks(range(3,11))
plt.grid(True)
plt.show()

# Interprétation des résultats (partie rapport)

#----Observation principale---- :
#la valeur d'inertia diminue lorsque le nombre de clusters (n_clusters) augmente
#En effet, plus il y a de clusters, plus les points sont proche de leurs centroide ce qui réduit la variance intra-cluster

#----Méthode du coude (Elbow Method)---- :
#On cherche le point oû la diminution de l'inertia devient moins importante
#Ce point correspond à un compromis entre un modèle simple (peu de clusters) et un modèle performant ( faible inertia)


#-> D'après le graphe, la baisse de l'inertia est très marquée entre 3 et 5 clusters,puis la courbe commence à s'aplatir. Cela indique la présence d'un "coude" autour de k=5

#----Choix du nombre de clusters ----:
#Le nombre optimal de clusters pour ce dataset semble donc être 5
#car ajouter davantage de clusters n'améliore que faiblement la qualité du groupement

#----Conclusion---- :
#La méthode du coude (Elbow Method) montre que la diminution de l'inertia ralentit après k=5
#Cela indique que 5 clusters offrent un bon compromis entre simplicité et qualité de regroupement
#Ainsi, pour le dataset "google_review_ratings.csv", le nombre optimal de clusters est de k=5
