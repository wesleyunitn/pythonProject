import numpy as np

from classes import Spettri
import funzioni as fun
import matplotlib.pyplot as plt
import seaborn as sns
from data_load import *

#CREO DUE ISTANZE DI PICCHI DAGLI STESSI DATI CON CONDIZIONI DIVERSE
picchi1 = Spettri(data1, npicchi = 30 ,prop={'height': (None, None), 'width': (None, None), 'prominence': 0.000001})
picchi1.normalizer()
picchi1.spettri = picchi1.peakfinder()

picchi1_nonum= Spettri(data1, npicchi=None,sortby='prominences',prop={'height': (None, None), 'width': (None, None), 'prominence': 0.000001})
picchi1_nonum.normalizer()
picchi1_nonum.spettri=picchi1_nonum.peakfinder()

X_nfix=fun.featextract1(picchi1.spettri,proplist=['peak_heights','prominences','K'])
X_nonum=fun.featextract1(picchi1_nonum.spettri,proplist=['peak_heights','prominences','K'], statlist=['mean','count','std'])

print(f'{X_nonum.head()}')
print(f'{X_nfix.head()}')

#PROCEDURE DI CLUSTERING

from sklearn.cluster import KMeans,DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler_nonum=StandardScaler()
X_scaled= scaler.fit_transform(X_nfix)
X_scaled_nonum = scaler_nonum.fit_transform((X_nonum))

pca = PCA(n_components=2)
pca_nonum = PCA(n_components=2)
X_pca= pca.fit_transform(X_scaled)
X_pca_nonum= pca_nonum.fit_transform(X_scaled_nonum)


# PER NUMERO DI PICC
listkmeans= []
for n in range(2,10):
    kmeans=KMeans(n_clusters=n,random_state=0)
    kmeans.fit(X_pca)
    print(f'{np.unique(kmeans.labels_,return_counts=True)}')
    print(f'con {n} clusters : {silhouette_score(X_pca, kmeans.labels_)}')
    listkmeans.append(kmeans)

yi = [arr.inertia_ for arr in listkmeans]
sns.lineplot(x=np.array(range(2,10)),y=yi, marker= 'o')

print(f"{[listkmeans[n].inertia_-listkmeans[n+1].inertia_ for n in range(len(listkmeans)-1)]}")
plt.show()
#le precedenti righe mi portano a scegliere un numero di picchi 6+-1

# # QUESTO CON NUMERI DI PICCHI NON FISSATI MA UTILIZZATI COME PARAMETRO
# listkmeans_nonum=[]
# for n in range(2,10):
#     kmeans=KMeans(n_clusters=n,random_state=0)
#     kmeans.fit(X_pca_nonum)
#     print(f'{np.unique(kmeans.labels_,return_counts=True)}')
#     print(f'con {n} clusters : {silhouette_score(X_pca, kmeans.labels_)}')
#     listkmeans_nonum.append(kmeans)
# #
# yi_nonum= [arr.inertia_ for arr in listkmeans_nonum]
# sns.lineplot(x=np.array(range(2,10)),y=yi_nonum, marker= 'o')
# plt.show()
# print(f"{[listkmeans_nonum[n].inertia_-listkmeans_nonum[n+1].inertia_ for n in range(len(listkmeans_nonum)-1)]}")
#le uprecedenti righe di codice portano a fissare i cluster di kmeans a 6

# PLOT PER VEDER I CLUSTER
# X_nonum['labels']=listkmeans_nonum[2 ].labels_
# # sns.scatterplot(data=X_nfix, x='K_mean',y='prominences_mean' ,hue= listkmeans_nonum[3].labels_)
# sns.pairplot(data=X_nonum,hue='labels',vars=['K_mean','K_std','prominences_mean','count','peak_heights_mean'])
# plt.show()

# X_nfix['labels']=listkmeans[5].labels_
# # sns.scatterplot(data=X_nfix, x='K_mean',y='prominences_mean' ,hue= listkmeans[3].labels_)
# sns.pairplot(data=X_nfix,hue='labels')
# plt.show()



# sns.pairplot(X_nfix,vars=['peak_heights_mean','prominences_mean','K_mean','K_std'])
# plt.show()
# sns.pairplot(X_nonum,vars=['count','prominences_mean','peak_heights_mean','peak_heights_std'])
# plt.show()
