from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd
from scipy.spatial.distance import pdist
import data_load
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from funzioni import index_translate
from sklearn.cluster import KMeans,DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.cluster import silhouette_score

class Peakfind(BaseEstimator, TransformerMixin):


    def __init__(self, npicchi=20, prop=None, cond=None, sortby='prominences'):
        """ prop : dizionario: proprietà e valori da passare a find_peaks, includo sempre almeno 'prominences' alla fase inizializzazione perchè senza limitare
        questo i picchi trovati non hanno molto significato

        sortby : string: i picchi trovati da find peaks tramite serie peak (chiamata anche in peakfinder) vengono riordinati in base a questa prorpietà selezionata (finale)
                   è meglio se non è mai None

        cond : float: soglia minima che i dati devono rispettare per essere mantenuti se il numero di picchi non è fissato (necessario che sortby non sia None in quel caso)

            """

        self.prop = prop
        self.npicchi = npicchi
        self.cond = cond
        self.sortby = sortby


    # # richiamato da TRANSFORM
    # def normalizer(self):
    #     for j in self.data.columns:
    #         if j != self.frq_col:
    #             try:
    #                 self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data[self.frq_col])
    #             except(KeyError):
    #                 self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data.iloc[:,0])
    #
    #     return self

    # HELPER SERIEPEAK
    def seriepeak(self, serie):
        # dictprop serve a passare argomenti alla funzione find_peaks
        """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà scelte per i picchi
          eto tutto il risultato di find_peaks con le proprietà specificate sull'attributoo di classe prop
          """
        # if self.prop is not None:
        #     peakobj = find_peaks(serie, **self.prop)
        # else:
        #     peakobj = find_peaks((serie))
        # DEVO MODIFICARE, NEL CASO PROP FOSSE NONE PER NON AVERE ERRORE
        peakobj = find_peaks(serie, **self.prop)
        da = pd.DataFrame(peakobj[1])
        da['peak_ind' + f'_{serie.name}'] = peakobj[0]

        if self.sortby is not None:
            da.sort_values(self.sortby, axis=0, ascending=False, inplace=True)

        if self.npicchi is not None:
            return da.iloc[:self.npicchi, :].reset_index(drop=True)
        elif (self.cond is not None) & (self.sortby is not None):
            # solo se cond e sortby non sono none! ritorna i picchi che per la proprietà indicata da sortby superano almeno una certa soglia data da cond
            return da[da[self.sortby] >= self.cond].reset_index(drop=True)

        else:
            return da.reset_index(drop=True)

    def fit(self, X, y=None, index = None, frq_col = 'K'):
        if index is None:
            self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        else:
            self.index = index
        if self.prop is None:
            self.prop = {'prominence': (None, None), 'height': (None, None), 'width': (None, None)}
        self.frq_col = frq_col
        print('fit fittizio')

        # dei check standard per stimatori custom
        X = check_array(X, accept_sparse=True,allow_nd=True)
        self.n_features_ = X.shape[1]
        # self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        """
        Accetta un campione di 121 spettri in forma di dataframe previamente normalizzato con normalizer
        ritorna una lista di dataframe con le peak_heights il numero d'onda e tutte le proprietà dei picchi che vengono
        dall utilizzo delle opzioni per la funzione find_peaks : ['height','width','prominence','distance','threshlod'..scipy.signal.find_peaks documentazione]
        Memorizza i picchi e le proprietà come una lista di 121 dataframe con lo stesso numero di colonne ed un numero di righe possibilmente variabile

        crea un attributo PICCHI per la lista di picchi
        """
        X_c = pd.DataFrame(X)
        #NORMALIZZAZIONE
        for j in X_c.columns:
            if j != self.frq_col:
                try:
                    X_c[j] = X_c[j] / integrate.trapz(X_c[j], x=X_c[self.frq_col])
                except(KeyError):
                    X_c[j] = X_c[j] / integrate.trapz(X_c[j], x=X_c.iloc[:, 0])


        if self.prop is None:
            self.prop = {'prominence': (None, None), 'height': (None, None), 'width': (None, None)}
        datalist_ = []
        for x in X_c.columns:

            if x != self.frq_col:
                temp = self.seriepeak(X_c[x])
                try:
                    temp[self.frq_col] = X_c.loc[:, self.frq_col][temp.iloc[:, -1].values].values
                except(KeyError):
                    temp[self.frq_col] = X_c.iloc[:,0][temp.iloc[:, -1].values].values
                datalist_.append(temp)
                del temp
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True, allow_nd=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return datalist_


class Featextract(BaseEstimator, TransformerMixin):

    def __init__(self, proplist=None, statlist=None, index = None):
        self.proplist = proplist
        self.statlist = statlist
        self.index = index

    def fit(self, X, y=None):
        y = None
        print('fit_fittizio')
        if self.statlist is None:
            self.statlist = ['mean', 'std']

        if self.proplist is None:
            self.proplist = ['peak_heights', 'prominences', 'widths', 'K']



        if self.index is None:
            self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        self.n_features_ = len(X)
        X = check_array(X, accept_sparse=True, allow_nd= True)
        self.is_fitted = True
        return self

    def transform(self, X, y=None):

        y = None

        if 'count' in self.statlist:
            self.statlist.remove('count')
            count = True
        else:
            count = False

        if sum([key in pd.DataFrame(X[0]).columns for key in self.proplist]) != len(pd.DataFrame(X[0]).columns):
            self.proplist = pd.DataFrame(X[0]).columns
            dic = {str(key) + '_' + str(stat): [] for key in self.proplist for stat in range(3)}
        else:
            dic = {str(key) + '_' + str(stat): [] for key in self.proplist for stat in self.statlist}
        #
        # if sum([key in pd.DataFrame(X[0]).columns for key in self.proplist]) == len(pd.DataFrame(X[0]).columns):
        #     dic = {key + '_' + stat: [] for key in self.proplist for stat in self.statlist}
        # else:
        #     dic = {str(key) + '_' + stat: [] for key in pd.DataFrame(X[0]).columns for stat in self.statlist}



        numpicchi = []  # utile solo con count=True
        for i in X:

            test = pd.DataFrame(i)
            try:
                test = test.iloc[:, 0:].describe().iloc[0:3, :] #test = test.get(list(self.proplist)).describe().loc[self.statlist, :]
            except(AttributeError):
                test = test.iloc[:,0:].describe().loc[self.statlist, :]
            except():
                test = test.iloc[:, 0:].describe().iloc[0:3, :]


            for x in self.proplist:
                try:
                    for k in self.statlist:
                        dic[str(x) + '_' + k].append(test.loc[k, x])
                except(KeyError):
                    for k in range(3):
                        dic[str(x) + '_' + str(k)].append(test.iloc[k, x])



            numpicchi.append(pd.DataFrame(i).get(list(self.proplist)).describe().loc['count'][0])

        for key in dic:
            dic[key] = pd.Series(dic[key], index=self.index[1:len(X)+1], name=key)

        if count:
            dic['count'] = numpicchi
            self.statlist.append('count')

        self.feat_ = pd.DataFrame(dic)
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=True,allow_nd= True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        # if len(X) != self.n_features_:
        #     raise ValueError('Shape of input is different from what was seen'
        #                      'in `fit`')
        return self.feat_

class Last_pipe_step(BaseEstimator, TransformerMixin):
    """ questa classe serve solo per tentare uno scoring per i cluster non informativi
    sulla base delle distanze fisiche medie dei cluster nella grigli 11X11"""

    def __init__(self,type = 'km',n_clusters= 6, eps = 0.4, min_samples = 4):
        """ eer
                      """
        self.type = type
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y = None):
        print('fit_fittizio')



        if self.type == 'db':
            self.cl_ = DBSCAN(eps = self.eps, min_samples= self.min_samples)
        elif self.type == 'km':
            self.cl_ = KMeans(n_clusters= self.n_clusters)

        self.cl_.fit(X)

        X = check_array(X, accept_sparse=True, allow_nd = True)
        self.is_fitted_ = True
        self.n_features_ = X.shape[1]
        return self


    def transform(self, X, y= None):

        return

    def score(self,X,y=None):
        df = pd.DataFrame(X, index=[f'row{i}col{j}' for i in range(1,12) for j in range(1,12)])
        df['labels'] = self.cl_.predict(X)
        print(df)
        self.clust_dist_ = []

        for i in np.unique(df['labels']):
            clusterpoint = index_translate(df[df['labels']== i].index)
            self.clust_dist_.append(np.mean(pdist(clusterpoint)))

        score = np.mean(self.clust_dist_)
        print(score)
        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Input validation
        X = check_array(X, accept_sparse=True, allow_nd= True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return score

    def predict(self,X,y=None):
        X = check_array(X, accept_sparse=True ,allow_nd=True)
        check_is_fitted(self, 'is_fitted_')
        return self.cl_.predict(X) #np.ones(X.shape[0], dtype=np.int64)








# p1 = Peakfind()
# picchi1 = p1.fit_transform(data_load.data1)
# fex1 = Featextract()
# feat_data1 = fex1.fit_transform(picchi1)
# print(feat_data1)
step = [('findpeak',Peakfind()),('feat',Featextract()),('scaler',StandardScaler()),('pca',PCA()),('db',DBSCAN())]
pipe = Pipeline(step)
pipe.fit(data_load.data1)
# ,('feat',Featextract())
print(pipe.transform(data_load.data1))

feat = Featextract()
find = Peakfind()
find.fit(data_load.data1)
# print(find.seriepeak(data_load.data1.iloc[:,4]))
# check_estimator(find)


# # #
params = {'findpeak__npicchi':[20,25,30]}
gsrch = GridSearchCV(pipe, param_grid = params, cv = 2 , scoring='silhouette_score')
gsrch.fit(data_load.data1)
print(gsrch.best_params_)

# from sklearn.model_selection import train_test_split
# Xtr,Xtes = train_test_split(data_load.data1)
# print(Xtes.isna().sum().sum())