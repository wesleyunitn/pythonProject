from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score
from sklearn.decomposition import PCA

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def km_cluster_plt(feat, pca: int = None, n_max_clusters=11, plot=False):
    """ fa un fit di (n_max_clusters -1) KMeans classifier con un numero di cluster da 2 a n_max_clusters
e plotta l'andamento del coefficente .inertia degli stessi per 'capire subito quale numero di clusters conviene'
    @param feat: attributo .feature della classe .Spettri()
    @param pca: numero di componenti da tenere per fare la pca if None non si fa la pca
    @param n_max_clusters: numero massimo di cluster da considerare per creare la lisrta di KMeans object,, sempre a partire da due
    -----------------------------------------------------------

    @return: lista di oggetti KMeans fittati
    """
    if pca > len(feat.columns):
        print('Non posso prendere più componenti del numero di feature dei dati da trasformare')
        pca = len(feat.columns)

    if pca is not None:
        scaler = StandardScaler()
        Pca = PCA(n_components=pca)
        Xpca = Pca.fit_transform(scaler.fit_transform(feat))
    else:
        Xpca = feat
    listkmeans = []
    for n in range(2, n_max_clusters):
        kmeans = KMeans(n_clusters=n, random_state=12)
        kmeans.fit(Xpca)
        # print(f'{np.unique(kmeans.labels_,return_counts=True)}')
        print(f'con {n} clusters : {silhouette_score(Xpca, kmeans.labels_)}')
        listkmeans.append(kmeans)

    # print(f"{[listkmeans[n].inertia_ - listkmeans[n + 1].inertia_ for n in range(len(listkmeans) - 1)]}")
    if plot:
        yi = [arr.inertia_ for arr in listkmeans]
        sns.lineplot(x=np.array(range(2, n_max_clusters)), y=yi, marker='o')
        plt.show()
    return listkmeans


def db_cluster_plt(feat, n_components=2, min_samples=5, eps=0.3, delta_search_multiplier=0.2, n_search=10):
    dblist = []
    for x in np.linspace(eps - delta_search_multiplier * eps, eps + delta_search_multiplier * eps, n_search):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components, random_state=12)
        featpca = pca.fit_transform(scaler.fit_transform(feat))
        db = DBSCAN(min_samples=min_samples, eps=x )
        db.fit(featpca)
        print(f'con eps =  {x} abbiamo {np.unique(db.labels_,return_counts=True)} clusters e lo score : {silhouette_score(featpca,db.labels_)}')

        dblist.append(db)

    # sns.scatterplot(x = featpca[:,0],y=featpca[:,1],hue = db.labels_)

    return dblist

#
#
# scaler = StandardScaler()
# scaler_nonum=StandardScaler()
# X_scaled= scaler.fit_transform(X_nfix)
# X_scaled_nonum = scaler_nonum.fit_transform((X_nonum))
#
# pca = PCA(n_components=2)
# pca_nonum = PCA(n_components=2)
# X_pca= pca.fit_transform(X_scaled)
# X_pca_nonum= pca_nonum.fit_transform(X_scaled_nonum)
#
#
# # PER NUMERO DI PICC
# listkmeans= []
# for n in range(2,10):
#     kmeans=KMeans(n_clusters=n,random_state=0)
#     kmeans.fit(X_pca)
#     print(f'{np.unique(kmeans.labels_,return_counts=True)}')
#     print(f'con {n} clusters : {silhouette_score(X_pca, kmeans.labels_)}')
#     listkmeans.append(kmeans)
#
# yi = [arr.inertia_ for arr in listkmeans]
# sns.lineplot(x=np.array(range(2,10)),y=yi, marker= 'o')
#
# print(f"{[listkmeans[n].inertia_-listkmeans[n+1].inertia_ for n in range(len(listkmeans)-1)]}")
# plt.show()
