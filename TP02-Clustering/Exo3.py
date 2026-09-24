#Exercice 3 : comparaison des performance K-means et Agglomerative Clustering sur Iris
#Objectif : Evaluer les performances de clustering avec l'Adjusted Rand Index pour différents méthodes de linkage

import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets, metrics

# Dataset Iris
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'])
y = pd.DataFrame(iris.target, columns=['Targets'])

# KMeans 
kmeans=KMeans(n_clusters=3, n_init=10,random_state=0)
kmeans.fit(X)

#Evaluation avec Adjusted Rand Index ( ARI)
ari_kmeans = metrics.adjusted_mutual_info_score(y.Targets, kmeans.labels_)
print(f"K- means Adjusted Rand Index : {ari_kmeans: .2f} \n ")

#Agglomerative Clustering
linkages=['single','complete','average','ward']
for link in linkages:
    cluster= AgglomerativeClustering(n_clusters=3,linkage=link)
    cluster.fit(X)
    ari=metrics.adjusted_rand_score(y.Targets, cluster.labels_)
    print(f"Agglomerative({link}) ADjusted Rand Index : {ari : .2f} \n")



# Interprétation des résultats (partie rapport)

#----K-means-----: 
#ARI = 0,76, très bonne correspondance avec les classes réelles 
#les clusters sont équilibrès et centrès autour des centroids

#----Agglomerative----- :
#Single linkage : ARI= 0.56, moins performant, sensible aux points isolées
#Complete linkage : ARI = 0.64, clusters plus compacts
#Average linkage : ARI = 0.76, correspondance optimale avec les classes réelles
#Ward linkage : ARI = 0.73, très performant, minimise la variance intra-cluster

#----Conclusion---- : 
#K-means et Agglomerative ( average et Ward) sont les méthodes les plus fiables pour retrouver les classes de l'Iris
#Single linkage est moins effiace à cause de la sensibilitée aux points extrêmes
 