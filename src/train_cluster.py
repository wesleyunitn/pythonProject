import pandas as pd
from sklearn.metrics import make_scorer
import data_load_notebook as datas
from classes import Spettri
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
import funzioni


def my_silhouette_score(estimator, X,y_test = None , y_true = None ):
     trsf = estimator.fit_predict(X)
     score = silhouette_score(X,trsf)
     return score

def my_pdist_score(estimator, X):
    labels = estimator.fit_predict(X)
    scores = funzioni.spatial_score(X, labels)
    score = np.mean([x for x in scores.values()])
    return score.dtype(float)


def my_davies_score(estimator, X, y_true = None):
    labels = estimator.fit_predict(X)

    score = davies_bouldin_score(X,labels)
    return -score

pk1 = Spettri(datas.data1)
pk1.peakfinder()
pk1.featextract()
pk1.featextract2()
# print(df[f'peak_obj_4'].feature.columns)

estimators_km = [('scaler',StandardScaler()),('pca',PCA()),('kmean',KMeans(random_state=00))]
estimators_db = [('scaler',StandardScaler()),('pca',PCA()),('dbscan',DBSCAN())]
pipekm = Pipeline(steps=estimators_km)
pipedb = Pipeline(steps = estimators_db)
#
scorer = make_scorer(my_silhouette_score,greater_is_better= False)
param = {'pca__n_components':[2,3,4,8],'kmean__n_clusters':[5,6,7,8,9]}
gsearchkm = GridSearchCV(pipekm,param_grid=param, scoring = my_davies_score, verbose= True, return_train_score= True)
gsearchkm.fit(pk1.feature)
print(gsearchkm.best_params_)

#