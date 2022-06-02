import pandas as pd

import data_load_notebook as datas
from classes import Spettri
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import silhouette_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate

#
# name = 'picchi'
# df = dict()
# for n,x in enumerate([10,20,30,40,50]):
#
#     df[f'peak_obj_{n}'] =  Spettri(datas.data1,npicchi=x)
#     df[f'peak_obj_{n}'].peakfinder()
#     df[f'peak_obj_{n}'].featextract()
#


# print(df[f'peak_obj_4'].feature.columns)
# scale  = StandardScaler()
# pca = PCA()
# db = DBSCAN()
# pca.fit(scale.fit_transform(df['peak_obj_2'].feature.values))
# labels = db.fit_predict(pca.transform(df['peak_obj_2'].feature.values))
# print(labels)
estimators_km = [('scaler',StandardScaler()),('pca',PCA()),('kmean',KMeans())]
pipekm = Pipeline(steps=estimators_km)

labels = pipekm.fit_predict(df['peak_obj_1'].feature.values)
#
# param = {'pca__n_components':[2,3,4,8],'kmean__n_clusters':[5,6,7,8,9]}
# gsearch = GridSearchCV(pipekm,param_grid=param)
# gsearch.fit(df['peak_obj_2'].feature.values)
# cross_validate(pipekm,df['peak_obj_2'].feature.values, scoring= 'davies_bouldin_score', cv =3)
# print(gsearch.best_params_)