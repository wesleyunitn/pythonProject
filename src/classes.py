from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd






class Spettri:

    def __init__(self, data, npicchi=15, prop=None, cond=None, sortby='prominences'):
        """ prop : dizionario: proprietà e valori da passare a find_peaks, includo sempre almeno 'prominences' alla fase inizializzazione perchè senza limitare
        questo i picchi trovati non hanno molto significato

        sortby : string: i picchi trovati da find peaks tramite serie peak (chiamata anche in peakfinder) vengono riordinati in base a questa prorpietà selezionata (finale)
                   è meglio se non è mai None

        cond : float: soglia minima che i dati devono rispettare per essere mantenuti se il numero di picchi non è fissato (necessario che sortby non sia None in quel caso)

            """
        if prop is None:
            prop = {'prominence': 3 * 10 ** -6, 'height': (None,None)}
        self.data = data
        self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        self.prop = prop
        self.npicchi = npicchi
        self.cond = cond
        self.sortby = sortby
        self.normalizer()  # lo aggiungo da subito dfato che la normalizzazione non è un opzione nel mio modello



    def seriepeak(self, serie):
        # dictprop serve a passare argomenti alla funzione find_peaks
        """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà scelte per i picchi
          e come ultima colonna l'indice corrispondente alla serie originale dei valori del picco:
          cond: float che indica la soglia minima da utilizzare come filtro sulla proprietà sortby per i picchi
          sortyby is not None---> colonna con cui la serie verrà riordinata
          npicchi is None---> si esegue una scelta sui picchi da tenere secondo  cond e sortby se entrambi sono diversi da None
          altrimenti, viene ritornato tutto il risultato di find_peaks con le proprietà specificate sull'attributoo di classe prop
          """

        peakobj = find_peaks(serie, **self.prop)
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

    def peakfinder(self):
        """
        Accetta un campione di 121 spettri in forma di dataframe previamente normalizzato con normalizer
        ritorna una lista di dataframe con le peak_heights il numero d'onda e tutte le proprietà dei picchi che vengono
        dall utilizzo delle opzioni per la funzione find_peaks : ['height','width','prominence','distance','threshlod'..scipy.signal.find_peaks documentazione]
        Memorizza i picchi e le proprietà come una lista di 121 dataframe con lo stesso numero di colonne ed un numero di righe possibilmente variabile

        crea un attributo PICCHI per la lista di picchi
        """
        datalist = []
        for x in self.data.columns:
            if x != 'K':
                temp = self.seriepeak(self.data[x])
                temp['K'] = self.data['K'][temp.iloc[:, -1].values].values
                datalist.append(temp)

        self.picchi = datalist

        return datalist

    def normalizer(self):
        for j in self.data.columns:
            if j != 'K':
                self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data.K)

        return self.data.head()


    def featextract(self,cols=('prominences','peak_heights','K'),statlist=('mean','std')):
        """ Metodo per estrarre dai picchi trovate le feature statistiche per le proprietà di interesse tra le colonne di self.picchi
            inoltre la lista di statistiche possibili è quella delle statistiche di pd.DaFtrame.describe (count è trattato in modo diverso dalle altre)"""
        if 'count' in statlist:
            statlist.remove('count')
            count = True

        else:
            count = False

        dic = {key + '_' + stat: [] for key in cols for stat in statlist}
        numpicchi = []  # utile solo con count=True
        for i in self.picchi:
            test = i
            test = test.get(list(cols)).describe().loc[statlist, :]
            for x in cols:
                for k in statlist:
                    dic[x + '_' + k].append(test.loc[k, x])

            numpicchi.append(i.get(list(cols)).describe().loc['count'][0])

        for key in dic:
            dic[key] = pd.Series(dic[key], index=self.index[1:], name=key)

        if count:
            dic['count'] = numpicchi

        return pd.DataFrame(dic)

    # def cluster_plot(self):