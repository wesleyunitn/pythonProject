from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd

import data_load


class Peakfind(BaseEstimator, TransformerMixin):

    def __init__(self, npicchi=20, prop=None, cond=None, sortby='prominences'):
        """ prop : dizionario: proprietà e valori da passare a find_peaks, includo sempre almeno 'prominences' alla fase inizializzazione perchè senza limitare
        questo i picchi trovati non hanno molto significato

        sortby : string: i picchi trovati da find peaks tramite serie peak (chiamata anche in peakfinder) vengono riordinati in base a questa prorpietà selezionata (finale)
                   è meglio se non è mai None

        cond : float: soglia minima che i dati devono rispettare per essere mantenuti se il numero di picchi non è fissato (necessario che sortby non sia None in quel caso)

            """
        self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        self.prop = prop
        self.npicchi = npicchi
        self.cond = cond
        self.sortby = sortby

    # richiamato da TRANSFORM
    def normalizer(self):
        for j in self.data.columns:
            if j != 'K':
                self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data.K)

        return self.data.head()

    # HELPER SERIEPEAK
    def seriepeak(self, serie):
        # dictprop serve a passare argomenti alla funzione find_peaks
        """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà scelte per i picchi
          eto tutto il risultato di find_peaks con le proprietà specificate sull'attributoo di classe prop
          """
        if self.prop is not None:
            peakobj = find_peaks(serie, **self.prop)
        else:
            peakobj = find_peaks((serie))
        # DEVO MODIFICARE, NEL CASO PROP FOSSE NONE PER NON AVERE ERRORE
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

    def fit(self, X, y=None):
        print('fit fittizio')
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
        for j in X_c.columns:
            if j != 'K':
                X_c[j] = X_c[j] / integrate.trapz(X_c[j], x=X_c.K)

        if self.prop is None:
            self.prop = {'prominence': (None, None), 'height': (None, None), 'width': (None, None)}
        datalist_ = []
        for x in X_c.columns:
            if x != 'K':
                temp = self.seriepeak(X_c[x])
                temp['K'] = X_c['K'][temp.iloc[:, -1].values].values
                datalist_.append(temp)

        return datalist_


class Featextract(BaseEstimator, TransformerMixin):

    def __init__(self, proplist=None, statlist=None, index = None):
        self.proplist = proplist
        self.statlist = statlist
        self.index = index

    def fit(self, X, y=None):
        print('fit_fittizio')
        return self

    def transform(self, X, y=None):

        if self.statlist is None:
            self.statlist = ['mean', 'std']

        if self.proplist is None:
            self.proplist = ['peak_heights', 'prominences', 'widths', 'K']

        if self.index is None:
            self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]

        if 'count' in self.statlist:
            self.statlist.remove('count')
            count = True
        else:
            count = False

        dic = {key + '_' + stat: [] for key in self.proplist for stat in self.statlist}
        numpicchi = []  # utile solo con count=True
        for i in X:
            test = i
            test = test.get(list(self.proplist)).describe().loc[self.statlist, :]
            for x in self.proplist:
                for k in self.statlist:
                    dic[x + '_' + k].append(test.loc[k, x])

            numpicchi.append(i.get(list(self.proplist)).describe().loc['count'][0])

        for key in dic:
            dic[key] = pd.Series(dic[key], index=self.index[1:], name=key)

        if count:
            dic['count'] = numpicchi
            self.statlist.append('count')
        return pd.DataFrame(dic)


p1 = Peakfind()
picchi1 = p1.fit_transform(data_load.data1)
fex1 = Featextract()
feat_data1 = fex1.fit_transform(picchi1)
print(feat_data1)
