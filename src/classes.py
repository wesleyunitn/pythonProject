from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd
from numpy.random import choice
''' 1 CLASSE : SPETTRI
'''


class Spettri():
    """
    Specializzata nel trattare i DATI CAMPIONE 11X11,ed eseguire estrazione picchi (.peakfinder) e la creazione di una statistica di questi ultimi (.featextract)

    """

    def __init__(self, data, npicchi=10, prop=None, cond=None, sortby='prominences'):
        """ ogni parametro viene memorizzato in un attributo dallo stesso nome

        :param prop: DIZIONARIO di proprietà e valori da passare a scipy.signal.find_peaks, inizializzata a non non limita la ricercain base a quei parametri

        :param sortby: string, i picchi trovati da find peaks tramite serie peak (chiamata anche in peakfinder) vengono riordinati in base a questa prorpietà selezionata (finale)
                   è meglio se non è mai None

        cond : float: soglia minima che i picchi dei dati devono rispettare per essere mantenuti se il numero di picchi non è fissato (necessario che sortby non sia None in quel caso)
        :param npicchi: int è il numero di picchi da estrarre
        :param data: i dati sotto forma di dataframe (come in data_load)
            """
        if prop is None:
            prop = {'height': (None,None),'prominence': (None,None), 'width':(None,None)}
        self.data = data
        self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        self.prop = prop
        self.npicchi = npicchi
        self.cond = cond
        self.sortby = sortby

          # lo aggiungo da subito dfato che la normalizzazione non è un opzione nel mio modello



    def __seriepeak__(self, serie):
        # dictprop serve UNICAMENTE (POCO COMPLETA) a passare argomenti alla funzione find_peaks, da cui il dunderscore
        """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà scelte per i picchi
          e come ultima colonna l'indice corrispondente alla serie originale dei valori del picco:
          cond: float che indica la soglia minima da utilizzare come filtro sulla proprietà sortby per i picchi
          sortyby is not None---> colonna con cui la serie verrà riordinata
          npicchi is None---> si esegue una scelta sui picchi da tenere secondo  cond e sortby se entrambi sono diversi da None
          altrimenti, viene ritornato tutto il risultato di find_peaks con le proprietà specificate sull'attributoo di classe prop

          :return : pd.DataFrame in cui ogni riga corrisponde un picco cui le corrispondenti proprietà ritornate da find_peaks() e una colonna
           di indici che lo mappa nella pd.series originale
          """

        peakobj = find_peaks(serie, **self.prop)
        # DEVO MODIFICARE, NEL CASO PROP FOSSE NONE PER NON AVERE ERRORE
        da = pd.DataFrame(peakobj[1], copy = True)
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

    def peakfinder(self, norm = True):
        """
        Accetta un campione di 121 spettri in forma di dataframe previamente normalizzato con normalizer
        ritorna una lista di dataframe con le peak_heights il numero d'onda e tutte le proprietà dei picchi che vengono
        dall utilizzo delle opzioni per la funzione find_peaks : ['height','width','prominence','distance','threshlod'..scipy.signal.find_peaks documentazione]
        Memorizza i picchi e le proprietà come una lista di 121 dataframe con lo stesso numero di colonne ed un numero di righe possibilmente variabile

        crea un attributo PICCHI come un dizionario di un dataframe in cui ogni dataframe corrisponde come chiave l'identificativo dello spettro (come negli indici di Spettri.data)
        """
        if norm == True:
            self.normalizer()

        datalist = []
        for x in self.data.columns:
            if x != 'K':
                temp = self.__seriepeak__(self.data[x])
                temp['K'] = self.data['K'][temp.iloc[:, -1].values].values
                datalist.append(temp)

        self.picchi = dict({key: datalist[n] for (n,key) in enumerate(self.index[1:])})


        return self.picchi

    def normalizer(self):
        self._norms = []
        for j in self.data.columns:
            if j != 'K':
                self._norms.append(integrate.trapz(self.data[j], x=self.data.K))
                self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data.K)

        return None


    def featextract(self,cols=('peak_heights','prominences','K','widths'),statlist=('mean','std')):
        """ Metodo per estrarre dai picchi trovate le feature statistiche per le proprietà di interesse tra le colonne di self.picchi
            inoltre la lista di statistiche possibili è quella delle statistiche di pd.DaFtrame.describe (count è trattato in modo diverso dalle altre)"""
        # self.normalizer()
        if 'count' in statlist:
            statlist.remove('count')
            count = True

        else:
            count = False

        dic = {key + '_' + stat: [] for key in cols for stat in statlist}
        numpicchi = []  # utile solo con count=True
        for i in self.picchi.values():
            test = pd.DataFrame(i, copy = True)  # per non cambiare il valore di i e quindi del dataframe di picchi iesimo per sbaglio
            test = test.get(list(cols)).describe().loc[statlist, :]
            for x in cols:
                for k in statlist:
                    dic[x + '_' + k].append(test.loc[k, x])

            numpicchi.append(i.get(list(cols)).describe().loc['count'][0])

        for key in dic:
            dic[key] = pd.Series(dic[key], index=self.index[1:], name=key)

        if count:
            dic['count'] = numpicchi

        self.feature = pd.DataFrame(dic, copy = True)
        return pd.DataFrame(dic, copy = True)




    def featextract2(self, prop = ['K','peak_heights','prominences','widths'], npeaks = None):
        ''' Sara utilizzabile solo se tutti gli spettri hanno lo stesso numero di picchi in (.picchi)
        IMPORTANTE E' SCONSIGLIATO  USARE NUMERI DI PICCHI TROPPO ALTI (OLTRE 10)

        npeaks : il numero di picchi da utilizzare per estrarre/creare le feature... dev essere minore o uguale al numero di picchi già trovati
        '''
        design_df = pd.DataFrame()
        if (npeaks is None)  :
            npeaks = self.npicchi
        for key in self.picchi.keys(): # passa in rassegna gli spettri
            ind = []
            l = []

            for n in range(npeaks):  # passa in rassegna le righe

                for x in prop:  # passa in rassegna le colonne
                    l.append(self.picchi[key][x][n])
                    ind.append(f'pk_{n+1}_{x}')

            toapp = {str: [val] for str,val in zip(ind,l)}

            design_df =  pd.concat([design_df, pd.DataFrame(toapp)] , ignore_index= True) # concatena le prop di ogni picco

        design_df.index = self.index[1:]
        self.feature2 = design_df

# class Database():
#
#
#     def __init__(self, database, npicchi=10, prop=None,sortby='prominences',cond=None):
#         self.data = database
#         if prop == None:
#             prop = {'height': (None,None), 'prominence' : (None,None), 'width' : (None,None)}
#         self.prop = prop
#         self.npicchi = npicchi
#         self.sortby = sortby
#         self.cond = cond
#
#     def __seriepeak__(self, serie):
#         # dictprop serve UNICAMENTE (POCO COMPLETA) a passare argomenti alla funzione find_peaks, da cui il dunderscore
#         """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà scelte per i picchi
#           e come ultima colonna l'indice corrispondente alla serie originale dei valori del picco:
#           cond: float che indica la soglia minima da utilizzare come filtro sulla proprietà sortby per i picchi
#           sortyby is not None---> colonna con cui la serie verrà riordinata
#           npicchi is None---> si esegue una scelta sui picchi da tenere secondo  cond e sortby se entrambi sono diversi da None
#           altrimenti, viene ritornato tutto il risultato di find_peaks con le proprietà specificate sull'attributoo di classe prop
#
#           :return : pd.DataFrame in cui ogni riga corrisponde un picco cui le corrispondenti proprietà ritornate da find_peaks() e una colonna
#            di indici che lo mappa nella pd.series originale
#           """
#
#         peakobj = find_peaks(serie, **self.prop)
#         # DEVO MODIFICARE, NEL CASO PROP FOSSE NONE PER NON AVERE ERRORE
#         da = pd.DataFrame(peakobj[1], copy=True)
#         da['peak_ind' + f'_{serie.name}'] = peakobj[0]
#
#         if self.sortby is not None:
#             da.sort_values(self.sortby, axis=0, ascending=False, inplace=True)
#
#         if self.npicchi is not None:
#             return da.iloc[:self.npicchi, :].reset_index(drop=True)
#         elif (self.cond is not None) & (self.sortby is not None):
#             # solo se cond e sortby non sono none! ritorna i picchi che per la proprietà indicata da sortby superano almeno una certa soglia data da cond
#             return da[da[self.sortby] >= self.cond].reset_index(drop=True)
#
#         else:
#             return da.reset_index(drop=True)





def plot_peaks(peak_obj: Spettri, nspettri=3, keys = None):
    ''' accetta un oggetto Spettri()) su cui è stata eseguita la ricerca dei picchi plotta un numero variabile
    di spettri scelti casualmente dal campione con relativi picchi gia memorizzati in Spettri.picchi'''
    ind = [f'row{n}col{m}' for n in range(1, 12) for m in range(1, 12)]
    # dfpicchi ={ key : peak_obj.picchi[x] for x,key in enumerate(ind)}
    fig, axes = plt.subplots(nspettri, 1, figsize=(12, 10))
    if keys is not None:
        chosens = keys
        nspettri = len(keys)
    else:
        chosens = choice(ind, nspettri)
    for n, keypeak in enumerate(chosens):
        axes[n].plot(peak_obj.data['K'], peak_obj.data[keypeak])
        axes[n].scatter(peak_obj.picchi[keypeak].K, peak_obj.picchi[keypeak]['peak_heights'], color='red')


def plot_spettri(peak_obj1: Spettri, peak_obj2: Spettri = None, nspettri=4, keys = None, same_ind= False, peaks = False):
    """ Plotta un numero fissato di spettri da uno o due oggetti spettri
    nspettri : numero di spettri per ogni campione da plottare --> numero di righe del subplot
    keys: lista di identificativi di spettri da selezionare per il plot ---> se None vengono scelti casualmente
    same_ind : se True utilizza gli stessi identificativi casuali o no per entrambi i plot ---> se keys è specificata viene comunque utilizzata per entrambi
    peaks : se True plotta anche i picchi trovati in precedenza
    """
    if peak_obj2 is not None:
        nobj = 2
    else:
        nobj = 1
    if keys is not None:
        chosen2 = list(keys)
        chosen1 = list(keys)
        print(chosen1)
        nspettri = len(keys)
    else:
        chosen1 = choice(peak_obj1.data.drop('K', axis=1).columns, nspettri)

    fig, axes = plt.subplots(nspettri, nobj, figsize=(20, 18), sharey=True)
    if nobj == 1:
        if nspettri == 1:
            axes.plot(peak_obj1.data['K'], peak_obj1.data[chosen1[0]], label=f'campione_1_spettro_{chosen1[0]}')
            axes.legend()
            if peaks:
                axes.scatter(peak_obj1.picchi[chosen1[0]]['K'], peak_obj1.picchi[chosen1[0]]['peak_heights'],color='red')
        else:
            for n in range(nspettri):
                axes[n].plot(peak_obj1.data['K'], peak_obj1.data[chosen1[n]], label=f'campione_1_spettro_{chosen1[n]}')
                axes[n].legend()
                if peaks:
                    axes[n].scatter(peak_obj1.picchi[chosen1[n]]['K'],peak_obj1.picchi[chosen1[n]]['peak_heights'], color = 'red')

    elif nobj == 2:
        if (same_ind is not None) & (keys is None):
            chosen2 = chosen1
        elif (same_ind is None) & (keys is None):
            chosen2 = choice(peak_obj2.data.drop('K', axis=1).columns, nspettri)

        for n in range(nspettri):
            axes[n, 0].plot(peak_obj1.data['K'], peak_obj1.data[chosen1[n]], label=f'campione_1_spettro_{chosen1[n]}')
            axes[n, 1].plot(peak_obj2.data['K'], peak_obj2.data[chosen2[n]], label=f'campione_2_spettro_{chosen2[n]}')
            if peaks:
                axes[n, 0].scatter(peak_obj1.picchi[chosen1[n]]['K'], peak_obj1.picchi[chosen1[n]]['peak_heights'], color='red')
                axes[n, 1].scatter(peak_obj2.picchi[chosen2[n]]['K'], peak_obj2.picchi[chosen2[n]]['peak_heights'], color='red')

            axes[n, 0].legend()
            axes[n, 1].legend()


def plot_database_peaks(datab, datab_peaks, nplot=3, keys = None):
    ''' Per plottare spettri del database con i picchi'''

    # plt.figure(figsize=(10,13))
    if keys is None:
        keysel = choice(list(datab.keys()), nplot)
    else:
        keysel = keys
        nplot = len(keys)
    fig, axes = plt.subplots(nplot, 1, figsize=(20, 25))
    if nplot == 1 :

        axes.plot(datab[keysel[0]]['K'], datab[keysel[0]]['H'], label=keysel[0])
        axes.scatter(datab_peaks[keysel[0]]['K'], datab_peaks[keysel[0]]['peak_heights'], color='red')
        axes.legend()

    else:
        for n, key in enumerate(keysel):
            axes[n].plot(datab[key]['K'], datab[key]['H'], label=key)
            axes[n].scatter(datab_peaks[key]['K'], datab_peaks[key]['peak_heights'], color='red')
            axes[n].legend()



