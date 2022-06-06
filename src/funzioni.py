from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist

names_col = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]


def normalizer(data):
    for j in data.columns:
        data[j] = data[j] / integrate.trapz(data[j], x=data.K)

    return data


def seriepeak(serie, npicchi=15, prop=None,sortby='peak_heights',cond=None):
    # dictprop serve a passare argomenti alla funzione find_peaks
    """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà dei picchi
      e come ultima colonna l'indice corrispondente alla serie originale dei valori del picco

      :param serie: è una serie scelta tra le colonne del dataframe nell'attributo data di questa classe (spettri)
      """
    if prop is None:
        prop={'height':2}

    peakobj = find_peaks(serie, **prop)

    da = pd.DataFrame(peakobj[1])
    da['peak_ind' + f'_{serie.name}'] = peakobj[0]

    #da1 = da.sort_values(sortby, axis=0, ascending=False)

    if npicchi!=None:
        return da.iloc[:npicchi, :].reset_index(drop=True)
    else:
        # da comlpetare per non fare il numero di picchi fissatiii
        return da[da[sortby]>=cond].reset_index(drop=True)

def peakfinder(data, n_picchi=15, prop=None,sortby='peak_heights'):
    """
    Accetta un campione di 121 spettri in forma di dataframe previamente normalizzato con normalizer
    ritorna una lista di dataframe con le peak_heights il numero d'onda e tutte le proprietà dei picchi che vengono
    dall utilizzo delle opzioni per la funzione find_peaks : ['height','width','prominence','distance','threshlod'..scipy.signal.find_peaks documentazione]

    """
    # il seguente dizionario servirà ad asssicurare che date delle proprietà, il sort-by sia scelto tra quelli possibili in bse alle proprietà selezionate
    dict_conversion= dict(height='peak_heights', prominence=['prominences', 'left_bases', 'right_bases'],
                          width=['widths','width_heights','prominences', 'left_bases', 'right_bases', 'peak_heights'],threshold=['left_thresholds','right_thresholds'])
    datalist = []
    for x in data.columns:
        if x != 'K':
            temp = seriepeak(data[x],npicchi= n_picchi, prop=prop,sortby=sortby)
            temp['K'] = data['K'][temp.iloc[:, -1].values].values
            datalist.append(temp)

    return datalist


def featextract1(datalist, statlist=tuple(['mean', 'std']), proplist=('peak_heights', 'prominences', 'K','widths'), index=tuple(names_col[1:])):
    """ trasforma la lista di dataframe con i vari picchi e proprietà per spettro in un
       dataframe unico con feature riassuntive per ogni spettro

      -statiist : lista di statistiche da utilizzare tra gli indici del risultato di describe
      -proplist : lista di proprietà dei picchi da tenere
      -index : è una lista di nomi legati ale posizioni fisiche degli spettri
      """
    if 'count' in statlist:
        statlist.remove('count')
        count = True
    else:
        count = False
    dic = {key + '_' + stat: [] for key in proplist for stat in statlist if stat}
    numpicchi = []
    for i in datalist:
        test = i
        test = test.get(list(proplist)).describe().loc[statlist, :]
        for x in proplist:
            for k in statlist:
                dic[x + '_' + k].append(test.loc[k, x])

        numpicchi.append(i.get(list(proplist)).describe().loc['count'][0])


    for key in dic:
        dic[key] = pd.Series(dic[key], index=index, name=key)

    if count:
        dic['count']=numpicchi

    return pd.DataFrame(dic)



def distpoint(labels_ordinated):
    """ plotta un griglia per visualizzare il campione 11X11 ragruppati in labels predette da un cluster
     :param LABELS : labels di un clustering che siano ----IMPO: ordinate secondo lo stesso schema unitlizzato dai campioni per le righe
     lo stesso che utilizzo negli script e cche non dovrebbe essere cambiato nell'eseguire gli stessi"""

    l = [(x, y) for x in range(1, 12) for y in range(1, 12)]
    grid = np.array(l).astype(float)
    plt.xlim(0,14)
    plt.ylim(0,14)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    sns.scatterplot(x = grid[:,0], y = grid[:,1] , hue = labels_ordinated)

    return None

def index_translate(index):
    """ traduce un alista di identificativi di uno spettro con unalista di punti in due dimensioni che ne mappa la posizione sulla grligra [in unità micron]

    @param index: lista di indici (string) del tipo (rowncolm for n,m in 1-11)
    @return: lista di punti come np.array(dtype=float)
    """
    temp=[]
    for x in index:
        temp.append(x[3:])
    points = []
    for y in temp:
        points.append((y[0],y[-1]))

    points = np.array(points).astype(float)
    return points


def spatial_score(feat_lab,labels_col= 'labels'):
    ''' accetta un dataframe tipo .feature di classe.Spettri() con una sola colonna di labels nominata come labels_col

    @param feat_lab:
    @param labels_col:
    @return:
    '''
    ris = dict()
    uniq_lab = np.unique(feat_lab[labels_col])
    for i in uniq_lab:
        index = index_translate(feat_lab[feat_lab[labels_col] == i].index)
        ris[f'cluster_{i}_dist'] = np.mean(pdist(index))
    return ris