import numpy as np
from sklearn.cluster import AgglomerativeClustering,KMeans

from collections import defaultdict


# Dataset Y et Z
Y = np.array([[1],[2],[4],[7],[8],[10],[15],[17],[21]])
Z = np.array([[11],[21],[22],[23],[26],[27],[28],[33],[35],[37],[47]])

labels_Y = ['1','2','4','7','8','10','15','17','21']
labels_Z = ['11','21','22','23','26','27','28','33','35','37','47'] 


#Fonction pour analyser un dataset

def analyze_dataset(data, labels, dataset_name, kmeans_init):
    print(f"\n DATASET {dataset_name} : ")
    
    #AgglomerativeClustering avec linkage single : 
    agg = AgglomerativeClustering(n_clusters=3,linkage='single')
    agg.fit(data)

    clusters_agg=defaultdict(list)
    for point_label, cluster_id in zip(labels, agg.labels_):
        clusters_agg[cluster_id].append(point_label)
    print("Agglomerative Clustering clusters:")
    for cid in sorted(clusters_agg.keys()):
        print(f"Cluster {cid}: {clusters_agg[cid]}")


    #Kmeans 
    #avec 3  clusters et centroids initiaux données
    kmeans= KMeans(n_clusters=3,init=kmeans_init,n_init=1)
    kmeans.fit(data)

    clusters_kmeans=defaultdict(list)

    for point_label,cluster_id in zip(labels,kmeans.labels_):
        clusters_kmeans[cluster_id].append(point_label)

    print("\nKMeans clusters :")
    for cid in sorted(clusters_kmeans.keys()):
        print(f"Cluster {cid} : {clusters_kmeans[cid]}")


kmeans_init_Y=np.array([[1],[2],[4]])
analyze_dataset(Y,labels_Y,"Y",kmeans_init=kmeans_init_Y)
kmeans_init_Z=np.array([[11],[21],[22]])
analyze_dataset(Z,labels_Z,"Z",kmeans_init=kmeans_init_Z)


#Interprétation des résultats (partie rapport)

#Observations principales :

#---Dataset Y----:
#Agglomerative : regroupe les points proches mais peut isoler les extrêmes ( exemple 21)
#K-means : forme des clusters plus équilibrès autour des centroids, intégrant les points extrêmes dans les clusters principaux
#Conclusion : les deux méthodes identifient les groupes naturels, k-means produit des clusters plus homogènes

#---Dataset Z----:
#Agglomerative : regroupe les points proche mais laisse certains points isolés (exemple 11,47)
#K-means : crée des clusters plus équilibrés, en intégrant ces point dans des groupes principaux (mais 11 isolé)
#Conclusion : Agglomerative montre les  points isolées, k-means forme des clusters plus réguliers

#---Conclusion Générale ---:
#Les deux méthodes de clustering présentent des comportements complémentaires :
#Agglomerative CLustering: fusionne les points les plus proches successivement (linkage='single'), ce qui permet de détecter les points isolées et la strcture naturelle des données.
#K-means : répartit la méjorité des points autour des centroids initiaux, recalculées à chaque itération, prosuisant des clusters homogènes et équilibrès.
#Le choix de la méthode dépend donc de l'objectif :
#   Pour visualiser la structure naruelle et détecter les points isolées -> Agglomerative
#   Pour obtenir des clusters homogènes et équilibrès -> K-means

