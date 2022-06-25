from src.classes import Spettri,plot_spettri,plot_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_load import *
from src import cluster_routine
import funzioni
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
#DATI CON CONDIZIONI DIVERSEes'],statlist=['mean','std','count']),pca=2)
import numpy as np
picchi2 = Spettri(data2)
picchi2.peakfinder()
picchi2.featextract()
picchi2.featextract2()

picchi1 = Spettri(data1)
picchi1.peakfinder()
plot_spettri(picchi2,picchi1,peaks=True)
# km2_list = cluster_routine.km_cluster_plt(picchi2.feature2, pca = 2)
# db2_list = cluster_routine.db_cluster_plt(picchi2.feature2, eps = 1, min_samples=3, n_components=3)
# km1_list = cluster_routine.km_cluster_plt(picchi1.feature, pca = 3,plot= True)
#
# cos1 = funzioni.spatial_score(picchi1.feature,km1_list[6].labels_)
# cos2 =  funzioni.spatial_score(picchi2.feature2, km2_list[3].labels_)
# param = {'pca__n_components':[2,3,4,8],'dbscan__eps': np.linspace(10**-3,1, 10)}
# gsearchdb = GridSearchCV(km2_list,param_grid=param, scoring = None)
# gsearchdb.fit(picchi2.feature)
# print(gsearchdb.best_params_)
