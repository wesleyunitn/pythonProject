from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd

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


def featextract1(datalist, statlist=tuple(['mean', 'std']), proplist=('peak_heights', 'prominences', 'K'), index=tuple(names_col[1:])):
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


# def distpoint(df)