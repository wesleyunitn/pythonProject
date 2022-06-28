import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cluster_routine
import funzioni
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import silhouette_score,davies_bouldin_score

from data_load import *
from classes import Spettri,plot_spettri,plot_peaks
import labelling
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score,make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV



pk1 = Spettri(data1)
pk1.peakfinder()
pk1.featextract()
pk1.featextract2()
print(f'picchi: {pk1.npicchi}  proprietà di ricerca : {pk1.prop}')

pk2 = Spettri(data2)
pk2.peakfinder()
pk2.featextract()

f1 = make_scorer(f1_score, average = 'macro')
#
# def f1(y_true,y_pred, *vars, **keyargs):
#     return f1_score(y_true, y_pred, *vars, average = None, **keyargs)

def train_gridsearch(pipe, X,y,cv, param_grid, scoring = f1 ):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    gsearch = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv= cv)
    gsearch.fit(X_train,y_train)
    predicted_total = gsearch.predict(X)
    recalled_spectre = X.loc[y==predicted_total,:].index
    print(f' gli spettri che il classificatore azzecca : {recalled_spectre}')
    return gsearch, recalled_spectre


# COMMENTED TO CALL THE MODULE
#
# skfold = StratifiedKFold(n_splits=4, random_state=1223, shuffle=True)
# log_reg = LogisticRegression(random_state= 100, multi_class='ovr', solver = 'sag', class_weight='balanced',max_iter=1000)
# steps = [('scaler', StandardScaler()), ('pca', PCA()),('log_reg',log_reg)]
# pipe = Pipeline(steps= steps)
# y_pk11 = list(labelling.wr11)
# X_pk11 = pd.DataFrame(pk1.feature, copy = True)
# param_grid = {'pca__n_components':[2,8], 'log_reg__C': [1, 0.7]}
# gsearc_fitted = train_gridsearch(pipe,X_pk11,y_pk11,skfold,param_grid, scoring = f1)
#
#
# skfold = StratifiedKFold(n_splits=4, random_state=1223, shuffle=True)
# log_reg = LogisticRegression(random_state= 100, multi_class='ovr', solver = 'sag', class_weight='balanced',max_iter=2000)
# steps = [('scaler', StandardScaler()), ('pca', PCA()),('log_reg',log_reg)]
# pipe = Pipeline(steps= steps)
#
# y = list(labelling.wr21)
# X = pd.DataFrame(pk2.feature, copy = True)
# param_grid = {'pca__n_components':[2,8], 'log_reg__C': [1, 0.7, 1,2]}
# gsearc_fitted, right_pred = train_gridsearch(pipe,X,y,skfold,param_grid, scoring = f1)


# gsearch = GridSearchCV(pipe, param_grid = param_grid, scoring= f1, cv = skfold )
# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=67)
# gsearch.fit(X_train,y_train)
# print(gsearch.score(X_test,y_test))

