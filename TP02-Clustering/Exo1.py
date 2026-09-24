#Exercice 1 : étude des méthodes de linkage avec le dendrogram
#Objectif : Comprendre comment fonctionnent les méthodes de linkage (single, complete , average, ward)  dans l'agglomerative Cluster en analysant le dendrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X= np.array([
    [2,2],   #A
    [3,3],   #B
    [10,10], #C
    [5,8],   #D
    [8,9],   #E
    [5,7],   #F
    [1,2]    #G
    
])

labels=['A','B','C','D','E','F','G']

#Les 4 méthodes de linkage à comparer : 
methods=['single','average','complete','ward']

plt.figure(figsize=(14,10))

for i,m in enumerate(methods) :
    #On applique un Agglomerative Clustering avec un linkage donné
    #Le linkage détermine comment la distance entre deux clusters est calculé lors des fusions successives
    linked = linkage(X,m)
    plt.subplot(2,2,i+1)
    plt.title(f"Dendrogram - {m} linkage ")

    dendrogram(linked,labels=labels,distance_sort='descending',show_leaf_counts=True)
plt.tight_layout()
plt.show()


#Interprétation des résultats (partie rapport)
#Observation importante :
#Les premières fusions concernent les points les plus proches du dataset,notamment A avec G et D avec F. Ces regroupements apparaissent dans toutes les méthodes, ce qui indique l’existence de clusters naturels.

#Cependant, l’ordre exact des fusions suivantes peut varier selon la méthode de linkage. Par exemple, B peut rejoindre le cluster {A,G} avant ou après la fusion de C et E. Cela s’explique par le fait que chaque méthode calcule différemment la distance entre clusters (distance minimale, moyenne, maximale ou variance).

#Malgré ces variations, trois groupes principaux se dégagent clairement : {A,B,G}, {D,F} et {C,E}.

#-----Single Linkage-----
#Après la formation des trois clusters, on observe que : 
#{D,F} rejoint rapidement {C,E} à une faible distance
#Puis ce grand groupe rejoint {A,B,G}
#Cela illustre bien le "chaining effect" caractéritique du single linkage,ou les clusters se relient progressivement par la distance minimale 

#-----Average Linkage---------
#Après la formation des trois clusters, on observe que : 
#Les mêmes groupes initiaux apparaissent, mais la fusion entre {D,F} et {C,E}  se fait à une distance plus grande par rapport au single linkage
#Les clusters sont donc plus équilibrès et moins étirés

#------Complete Linkage--------
#Après la formation des trois clusters, on observe que : 
#On observe que {A,B,G} reste séparé très longtemps des deux autres groupes
#La fusion finale se fait à une distance plus élevée
#Cela montre que complete linkage produit des clusters compacts et bien séparès


#------Ward Linakge----------
#Après la formation des trois clusters, on observe que : 
#Le comportement est similaire à complete linkage, mais encore plus marqué
#Les trois clusters naturels restent séparès le plus longtemps possible
#Ward minimise la variance intra-cluster ce qui donne le dendrogram le plus structuré


#------Conclusion------ 
#L'analyse des dendrigrammes montre clairement que le choix du linkage influence directeemnt la manière dont les clusters sont formés et la hauteur des fusions onservées.
#Cette hauteur correspond au mode de calcul de la distance entre clusters(minimale,maximale,moyenne ou basée sur la variance)
#Ainsi, le single linkage relie rapidmeent les groupes par effet de chaînage,tandis que le complete et le ward linkage mantienne des clusters compacts et le plus long possible. L'average linkage adopte un comportmeent intermédiaire

#Le dendrogramme appraît donc comme une représenation graphique fidèle de la règle mathémtique utilisée par chaque méthode de linkage, permettant de comprednre visuellement le processus de formation des clusters.

