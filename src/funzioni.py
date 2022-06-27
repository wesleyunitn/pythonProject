from scipy.signal import find_peaks
from scipy import integrate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist

names_col = ['K'] + [f'row{i}col{j}' for i in range(1, 12) for j in range(1, 12)]


def normalizer(y,x):
    y = y/integrate.trapz(y = y,x= x)
    return  y/integrate.trapz(y = y,x= x)


def seriepeak(serie, npicchi=10, prop=None,sortby='prominences',cond=None):
    # dictprop serve a passare argomenti alla funzione find_peaks
    """prende in input una serie-spettro e ritorna un DataFrame con diverse colonne a seconda delle proprietà dei picchi
      e come ultima colonna l'indice corrispondente alla serie originale dei valori del picco

      :param serie: è una serie scelta tra le colonne del dataframe nell'attributo data di questa classe (spettri)
      :param npicchi: numero di picchi da tenere, se None tiene tutti i picchi trovati con le condizioni
      """

    if prop is None:  # utile solo se viene direttamente chiamata seriepeak
        prop={'height': (None,None), 'prominence' : (None,None), 'width' : (None,None)}

    peakobj = find_peaks(serie, **prop)
    # CONTIENE LE PROPRIETà DEI PICCHI
    da = pd.DataFrame(peakobj[1])
    # E GLI INDICI
    da['peak_ind' + f'_{serie.name}'] = peakobj[0]

    # da1 = da.sort_values(sortby, axis=0, ascending=False)
    if sortby is not None:
        da.sort_values(by=sortby, inplace= True, ascending= False)
    if (npicchi is not None) & (sortby is not None):
        return da.iloc[:npicchi, :].reset_index(drop=True)
    else:
        # SISTEMO PER IL CASO CONDIZIONALE
        return da[da[sortby]>=cond].reset_index(drop=True)

#checked1
def peakfinder(database, prop = None, drop_ind = False, npicchi = 10, sortby = 'prominences', cond = None, norm = True):
    ''' analoga al metod della classe Spettri
    '''
    if prop == None :
        prop = {'prominence': (None, None), 'height': (None, None),'width': (None,None), 'wlen': 50}
    dtbase_peaks = dict()
    for key, val in database.items():
        if norm :
            val['H']= normalizer(val['H'], val['K'])
        # ora chiama la funzione seriepeak che effettivamente trova i picchi per uno spettro
        dtbase_peaks[key] = seriepeak(val['H'], prop= prop, npicchi = npicchi, cond = cond, sortby = sortby )
        # AGGIUNGO LA COLONNA CON I RISPETTIVI NUMERI D'ONDA
        dtbase_peaks[key]['K'] = val.loc[dtbase_peaks[key]['peak_ind_H'], 'K'].values
        if drop_ind:
            dtbase_peaks[key].drop('peak_ind_H', axis = 1, inplace = True)

    return dtbase_peaks

# def peakfinder(data, n_picchi=10, prop=None,sortby='peak_heights'):
#     """ FORMA PER IL DATABASE
#     """
#     # il seguente dizionario servirà ad asssicurare che date delle proprietà, il sort-by sia scelto tra quelli possibili in bse alle proprietà selezionate
#     dict_conversion= dict(height='peak_heights', prominence=['prominences', 'left_bases', 'right_bases'],
#                           width=['widths','width_heights','prominences', 'left_bases', 'right_bases', 'peak_heights'],threshold=['left_thresholds','right_thresholds'])
#     datalist = []
#     for x in data.columns:
#         if x != 'K':
#             temp = seriepeak(data[x],npicchi= n_picchi, prop=prop,sortby=sortby)
#             temp['K'] = data['K'][temp.iloc[:, -1].values].values
#             datalist.append(temp)
#
#     return datalist


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


def featextract1_df(dataf, statlist=tuple(['mean', 'std']), proplist=('peak_heights', 'prominences', 'K','widths'), index=tuple(names_col[1:])):
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
    for i in dataf:
        test = dataf[i]
        test = test.get(list(proplist)).describe().loc[statlist, :]
        for x in proplist:
            for k in statlist:
                dic[x + '_' + k].append(test.loc[k, x])

        numpicchi.append(dataf[i].get(list(proplist)).describe().loc['count'][0])


    for key in dic:
        dic[key] = pd.Series(dic[key], index=dataf.keys(), name=key)

    if count:
        dic['count']=numpicchi

    return pd.DataFrame(dic)


def featextract2_df(picchif, prop = None ):
    ''' Sara utilizzabile solo se tutti gli spettri hanno lo stesso numero di picchi in (.picchi)
           IMPORTANTE E' SCONSIGLIATO  USARE NUMERI DI PICCHI TROPPO ALTI (OLTRE 10)'''
    design_df = pd.DataFrame()
    if prop is None:
        prop = ['K','peak_heights','prominences', 'widths']
    for key in picchif.keys():  # passa in rassegna gli spettri
        ind = []
        l = []

        for n in range(len(picchif[key].iloc[:,0])):  # passa in rassegna le righe

            for x in prop:  # passa in rassegna le colonne
                l.append(picchif[key][x][n])
                ind.append(f'pk_{n + 1}_{x}')

        toapp = {str: [val] for str, val in zip(ind, l)}

        design_df = pd.concat([design_df, pd.DataFrame(toapp)], ignore_index=True)  # concatena le prop di ogni picco

    design_df.index = picchif.keys()
    return design_df


# FINE SEZIONE FUNZIONI ANALOGH AI METODI PER LA CLASSE sPETTRI
#
#
#
#










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
    ind = list(index)
    for x in ind:
        temp.append(x[3:])
    points = []
    for y in temp:
        points.append((y[0],y[-1]))

    points = np.array(points).astype(float)
    return points


def spatial_score(feat_lab,labels_col):
    ''' accetta un dataframe tipo .feature di classe.Spettri()

    @param feat_lab:
    @param labels_col:
    @return:
    '''
    ris = dict()
    feat  = pd.DataFrame(feat_lab,copy = True)
    feat['labels']= labels_col
    uniq_lab = np.unique(feat['labels'])
    for i in uniq_lab:
        index=[]
        index = index_translate(feat[feat['labels'] == i].index)
        ris[f'cluster_{i}_dist'] = np.mean(pdist(index))


    return ris