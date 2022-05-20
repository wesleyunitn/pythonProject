from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd


class Spettri:

    def __init__(self, data, npicchi=15, prop=None, cond=None, sortby='prominences'):
        if prop is None:
            prop = {'prominence': 3 * 10 ^ -4}
        self.data = data
        self.index = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]
        self.prop = prop
        self.npicchi = npicchi
        self.cond = cond
        self.sortby = sortby
    #
    # def makepeaks_list(self):
    #     """ metodo per creare la lista di dataframe
    #
    #     :param npicchi:
    #     :return:
    #     """
    #     self.spettri = self.peakfinder()

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

        da = pd.DataFrame(peakobj[1])
        da['peak_ind' + f'_{serie.name}'] = peakobj[0]

        if self.sortby is not None:
            da.sort_values(self.sortby, axis=0, ascending=False, inplace=True)

        if self.npicchi is not None:
            return da.iloc[:self.npicchi, :].reset_index(drop=True)
        elif (self.cond is not None) & (self.sortby is not None):
            # da comlpetare per non fare il numero di picchi fissatiii
            return da[da[self.sortby] >= self.cond].reset_index(drop=True)

        else:
            return da.reset_index(drop=True)

    def peakfinder(self):
        """
        Accetta un campione di 121 spettri in forma di dataframe previamente normalizzato con normalizer
        ritorna una lista di dataframe con le peak_heights il numero d'onda e tutte le proprietà dei picchi che vengono
        dall utilizzo delle opzioni per la funzione find_peaks : ['height','width','prominence','distance','threshlod'..scipy.signal.find_peaks documentazione]

        """
        datalist = []
        for x in self.data.columns:
            if x != 'K':
                temp = self.seriepeak(self.data[x])
                temp['K'] = self.data['K'][temp.iloc[:, -1].values].values
                datalist.append(temp)

        return datalist

    def normalizer(self):
        for j in self.data.columns:
            if j != 'K':
                self.data[j] = self.data[j] / integrate.trapz(self.data[j], x=self.data.K)

        return self.data.head()
