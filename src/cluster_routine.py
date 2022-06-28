from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import silhouette_score,davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,make_pipeline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import funzioni
import  labelling

def km_cluster_plt(feat, pca: int = None, n_max_clusters=(2,11), plot=False):
    """ fa un fit di (n_max_clusters -1) KMeans classifier con un numero di cluster da 2 a n_max_clusters
e plotta l'andamento del coefficente .inertia degli stessi per 'capire subito quale numero di clusters conviene'
    @param feat: attributo .feature della classe .Spettri()
    @param pca: numero di componenti da tenere per fare la pca if None non si fa la pca
    @param n_max_clusters: numero massimo di cluster da considerare per creare la lisrta di KMeans object,, sempre a partire da due
    @param plot: Bool: parametro per fare il plot
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
        print(f'{Pca.explained_variance_ratio_}')
    else:
        Xpca = feat
    listkmeans = []
    for n in range(n_max_clusters[0],n_max_clusters[1]):
        kmeans = KMeans(n_clusters=n, random_state=12)
        kmeans.fit(Xpca)
        # print(f'{np.unique(kmeans.labels_,return_counts=True)}')
        print(f'con {n} clusters, davies_bouldin_score : {davies_bouldin_score(Xpca, kmeans.labels_):.2f}')
        print(f'con {n} clusters, silohuette_score : {silhouette_score(Xpca, kmeans.labels_):.2f}')
        listkmeans.append(kmeans)

    # print(f"{[listkmeans[n].inertia_ - listkmeans[n + 1].inertia_ for n in range(len(listkmeans) - 1)]}")
    if plot:
        yi = [arr.inertia_ for arr in listkmeans]
        sns.lineplot(x=np.array(range(n_max_clusters[0], n_max_clusters[1])), y=yi, marker='o')
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
        print(f'con eps= {x:.2E} abbiamo {np.unique(db.labels_, return_counts=True)[1]} clusters e davies_bouldin_score : {davies_bouldin_score(featpca,db.labels_):.2f}')
        print(f' {np.unique(db.labels_,return_counts=True)[1]} clusters e silhouette_score : {silhouette_score(featpca,db.labels_):.2f}')

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

def clustering1(pk_obj, database, cluster_estimator, real_labels1, real_labels2 = None, pca_components1 = 2, pca_components2 = 2 , **clust_init):
    ''' per fare un custering automatico, con dei parametri fissati (hyperparameters TUNED someway)
        E CONFRONTARLO con delle labels esistenti

     Non è molto elastico, l'analisi è circa la stessa e non fa uso di uno scoring supervised
     ritorna pero entrambe le pipeline fittate per riprodurre evntualmente i risultati

     :cluster_cluster_estimator: uno stimatore di sklear.cluster (CALLABLE) come KMeans o DBSCAN
     :real_labels1. label dei daticon cui confrontare il clustering
     '''

    database_picchi = funzioni.peakfinder(database,prop=pk_obj.prop, npicchi = pk_obj.npicchi, sortby = pk_obj.sortby)
    database_feature = funzioni.featextract1_df(database_picchi)
    database_feat2 = funzioni.featextract2_df(database_picchi)

    col_ranges1 = [(min(pk_obj.feature[arr]),max(pk_obj.feature[arr])) for arr in pk_obj.feature.columns  ]
    col_ranges2 = [(min(pk_obj.feature2[arr]), max(pk_obj.feature2[arr])) for arr in pk_obj.feature2.columns]
    df_cols1 = {key:key for key in pk_obj.feature.columns}
    df_cols2 = {key:key for key in pk_obj.feature2.columns}
    pipe_feat1 = make_pipeline(StandardScaler(),PCA(pca_components1),cluster_estimator(**clust_init))
    pipe_feat2 = make_pipeline(StandardScaler(),PCA(pca_components2),cluster_estimator(**clust_init))

    # CLUSTER PREDETTI
    cluster_pred1 = pipe_feat1.fit_predict(pk_obj.feature)
    cluster_pred2 = pipe_feat2.fit_predict(pk_obj.feature2)
    # MATERIALI PRESENTI NELLE LABELS
    unique_labels1 = np.unique(real_labels1)
    if real_labels2 is not None:
        unique_labels2 = np.unique(real_labels2)
    else:
        unique_labels2 = unique_labels1
    # PREDIZIONI DEGLI SPETTRI DEI MATERIALI UNICI (DA LABELS)
    feat1_clusterpred_database_unique = pipe_feat1.predict(labelling.weighthed_scale(database_feature.loc[unique_labels1,pk_obj.feature.columns],col_ranges1,**df_cols1))
    feat2_clusterpred_database_unique = pipe_feat2.predict(labelling.weighthed_scale(database_feat2.loc[unique_labels2,pk_obj.feature2.columns],col_ranges2,**df_cols2))
    # STABILIAMO UN DIZIONARIO PER LA CONVERSIONE DEI CLUSTER IN IDENTIFICATORI DI MATERIALI
    map_dict1= {key:val for key,val in zip( unique_labels1, feat1_clusterpred_database_unique)}
    map_dict2 = {key: val for key, val in zip( unique_labels2, feat2_clusterpred_database_unique)}


    material_pred1 = np.array(converter(map_dict1,cluster_pred1))
    material_pred2 = np.array(converter(map_dict2, cluster_pred2))

    print(f' i correttamente predetti tramite feat1 : {np.sum(material_pred1== real_labels1)}'
          f'    i correttamente predetti tramite feat2 : {np.sum(material_pred2== real_labels1)}')
    return material_pred1, material_pred2, pipe_feat1, pipe_feat1


def converter(dict, numeric_list):
    ''' accetta un dizionario con valori numerici e indici stringhe ed una lista di identificativi numerici da trasformare secondo le corrispondenze del dizionario '''
    converted = []
    for n in range(len(numeric_list)):
        for k in dict.keys():

            if dict[k] == numeric_list[n]:
                converted.append(k)
                break

    return converted




