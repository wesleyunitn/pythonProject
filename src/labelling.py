import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import data_load as datas
from classes import Spettri, plot_spettri, plot_database_peaks
import funzioni
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)


def euclidean_weighted_arr(arr1,arr2,W = None):
    '''
    PER ARRAYYYYY 1D

    @param arr1:
    @param arr2:
    @param W:
    @return:
    '''
    if W is None:
        W = np.ones(arr2.shape)

    if (len(arr1)!=len(arr2))  | (len(arr1)!=len(W)):
        print('PROBLEMAPROBLEMAPROBLEMA')
        return

    else:

        sum = 0
        for n in range(len(arr1)):
            sum+= W[n]*(arr1[n]-arr2[n])**2
        ris = np.sqrt(sum)
        del sum
        return ris


def find_similar_material(spettrofeat, database, spettro='row1col1', printa=True, W = None):
    ''''dato un oggetto Spettro.feature o Spettro.feature2 e un oggetto analogo Database_feat1  [----> non è un attributo della classe Spettri]
    confronta UNA sola riga del primo con tutte quelle del secondo e trova la più simiile

    return: -dist_dic: dizionario con le distanze per ogni materiale,
            -similar_material : nome del materiale più simile
            -min : la distanza minima ----- per averla in modo veloce

    '''
    dist_dic = dict()
    for key in database.index:
        dist_dic[key] = euclidean_weighted_arr(spettrofeat.loc[spettro, :], database.loc[key, :], W)
    min = np.min(list(dist_dic.values()))
    index = list(dist_dic.values()).index(min)
    if printa:
        print(f'+ simile = {list(dist_dic.keys())[index]} con {min}')

    similar_material = list(dist_dic.keys())[index]
    return dist_dic, similar_material, min


def wrap(spettrofeat, database, W = None, printa = False):
    ''' wrapper di find_similar_material con tutto un database di spettri'''
    labels = []
    scores = []
    for key in spettrofeat.index:
        _j, label, score = find_similar_material(spettrofeat, database, W = W,  spettro=key, printa=printa)
        scores.append(score)
        labels.append(label)

    return np.array(labels), np.array(scores)


def weighthed_scale(df_to_transf,ranges, second_df_to_transf = None, save = False, **columns_bunch):
    '''  Funzione per riscalare le colonne di un dataframe con MinMax velocemente



    :second_second_df_transf: se specificato un secondo dataframe con nomi delle colonne UGUALI, trasforma anchesso, ma SENZA refittare i trasformatori precedentemente usati
    :ranges: lista di tuple che specifichi i range con cui riscalare i diversi gruppi di colonne
    : columns_bunch : DIZIONARIO o KEYWORD ARGUMENTS ogniuna con un nome significativo per il gruppo di colonne
     assegnate ad una lista di identificativi delle colonne del df da riscalare

    @note
    non serve  trasformare tutte le colonne del dataframe
    ma non se ne può trasformare una sola ------ credo sia dovuto proprio a Min Max scaler
    '''
    transformers = []
    bunch_names = list(columns_bunch.keys())
    cols_flattened = []


    if len(ranges) != len(columns_bunch): # minimo degug
        print('PROBLEMAPROBLEMAPROBLEMA')


    # PROVO A MANTENERE LA MEDIA PER TENERE LE CARTTERISTICHESEMPRE CENTRATE NELLO STESSO PUNTO MA 'STIRATE'
    # for n in range(len(ranges)):
    #     temp_diff = ranges[n][1]-ranges[n][0]
    #     mean = np.mean(df_to_transf[columns_bunch.keys()[n]])
    #

    for n,cols in enumerate(columns_bunch.values()):
        nametemp = bunch_names[n]
        scalertemp = MinMaxScaler(feature_range=ranges[n])
        transformers.append(tuple([nametemp,scalertemp,cols]))
        cols_flattened += cols
        del nametemp,scalertemp
    coltransf = ColumnTransformer(transformers = transformers)
    # df_transf = pd.DataFrame(coltransf.fit_transform(df_to_transf), columns = cols_flattened, index = df_to_transf.index)
    col_left = set(df_to_transf.columns)-set(cols_flattened)
    df_transf = pd.DataFrame(df_to_transf, columns=col_left, index=df_to_transf.index)
    df_transf[cols_flattened] = pd.DataFrame(coltransf.fit_transform(df_to_transf), index=df_to_transf.index)

    if second_df_to_transf is not None :
        second_df_transf = pd.DataFrame(second_df_to_transf, columns = col_left, index = second_df_to_transf.index)
        second_df_transf[cols_flattened] = pd.DataFrame(coltransf.transform(second_df_to_transf), index = second_df_to_transf.index)
        return df_transf,second_df_transf
    else:
        return df_transf




def static_plot_compare_labels(peak_obj, df_feature, labels, dir_title):
    ''' Serve a valutare più velocemente l'efficacia del labeling
    e crea una raccolta di plot dei picchi con un labeling sui materialio del database
    '''

    dfcopy = pd.DataFrame(df_feature, copy = True)
    dfcopy['labels'] = labels
    unique = np.unique(labels)
    os.makedirs(f'{dir_title}')
    for i in unique:
        if len(dfcopy[ dfcopy['labels'] == i].index) >= 10:
            os.makedirs(f'.\{dir_title}\{i}')
            nfig = (len(dfcopy[ dfcopy['labels'] == i].index) // 10) + 1
            spetr_per_fig = len(dfcopy[ dfcopy['labels'] == i].index) // nfig
            pos = 0
            for j in range(nfig):
                plot_spettri(peak_obj, peaks=True, keys=dfcopy[dfcopy['labels'] == i].index[pos: pos+spetr_per_fig])
                pos += spetr_per_fig
                plt.savefig(f'.\{dir_title}\{i}\{i}_{j+1}')
                plt.close()
        plot_spettri(peak_obj, peaks = True,  keys = dfcopy[ dfcopy['labels'] == i].index)
        plt.savefig(f'.\{dir_title}\{i}')
        plt.close()



#  le righe di codice sotto sono il prototipo con cui hio geerato le labels
# commentate per poter chiamare il modulo da notebook
#
#
# pk1 = Spettri(datas.data1)
# pk2 = Spettri(datas.data2)
#
# pk1.go()
# pk2.go()
#
# database_picchi = funzioni.peakfinder(datas.database, prop = pk1.prop)
# database_feat1 = funzioni.featextract1_df(database_picchi)
# database_feat2 = funzioni.featextract2_df(database_picchi)
# #
#
# # scaled_pk1_feat1 = weighthed_scale(pk1.feature,[(100,1000),(0,600),(0,400)],K_mean = ['K_mean'], p_h = ['prominences_mean','peak_heights_mean','K_50%'], std_s= ['prominences_std','K_std','peak_heights_std'])
# # scaled_pk1_feat2 = weighthed_scale(pk1.feature2,[(100,1000),(0,600),(0,400),(0,200)],pk_K = [f'pk_{n}_K' for n in range(1,11)], pk_hei = [f'pk_{n}_peak_heights' for n in range(1,11)], pk_pro= [f'pk_{n}_prominences' for n in range(1,11)], pk_widths = [f'pk_{n}_widths' for n in range(1,11)] )
# #
# # scaled_pk2_feat1 = weighthed_scale(pk2.feature,[(100,1000),(0,600),(0,400)],K_mean = ['K_mean'], p_h = ['prominences_mean','peak_heights_mean','K_50%'], std_s= ['prominences_std','K_std','peak_heights_std'])
# # scaled_pk2_feat2 = weighthed_scale(pk2.feature2,[(100,1000),(0,600),(0,400),(0,200)],pk_K = [f'pk_{n}_K' for n in range(1,11)], pk_hei = [f'pk_{n}_peak_heights' for n in range(1,11)], pk_pro= [f'pk_{n}_prominences' for n in range(1,11)], pk_widths = [f'pk_{n}_widths' for n in range(1,11)] )
# #
# #
# # scaled_database_feat2 = weighthed_scale(database_feat2,[(100,1000),(0,600),(0,400),(0,200)],pk_K = [f'pk_{n}_K' for n in range(1,11)], pk_hei = [f'pk_{n}_peak_heights' for n in range(1,11)], pk_pro= [f'pk_{n}_prominences' for n in range(1,11)], pk_widths = [f'pk_{n}_widths' for n in range(1,11)])
# # scaled_database_feat1 = weighthed_scale(database_feat1,[(100,1000),(0,600),(0,400)],K_mean = ['K_mean'], p_h = ['prominences_mean','peak_heights_mean','K_50%'], std_s= ['prominences_std','K_std','peak_heights_std'])
#
# wr11,score11 = wrap(pk1.feature,database_feat1)
# wr12,score12 = wrap(pk1.feature2,database_feat2)
#
# wr21,score21 = wrap(pk2.feature,database_feat1)
# wr22, score22 = wrap(pk2.feature2,database_feat2)
#
#
# print(np.sum(wr11==wr12))
# print(np.sum(wr21==wr22))

# wr11_scaled, score11_scaled = wrap(scaled_pk1_feat1, scaled_database_feat1)
# wr12_scaled,score12_scaled = wrap(scaled_pk1_feat2, scaled_database_feat2)
#
# wr21_scaled, score21_scaled = wrap(scaled_pk2_feat1, scaled_database_feat1)
# wr22_scaled,score22_scaled = wrap(scaled_pk2_feat2, scaled_database_feat2)
#
# # #
# static_plot_compare_labels(pk1,pk1.feature,wr11, 'pk1_feat1')
# static_plot_compare_labels(pk1,pk1.feature2,wr12,'pk1_feat2')
#
# static_plot_compare_labels(pk2,pk2.feature,wr21,'pk2_feat1')
# static_plot_compare_labels(pk2,pk2.feature2,wr22,'pk2_feat2')
# #
# static_plot_compare_labels(pk1,pk1.feature,wr11_scaled,'pk1_feat1_scaled')
# static_plot_compare_labels(pk1,pk1.feature2,wr12_scaled,'pk1_feat2_scaled')
#
# static_plot_compare_labels(pk2,pk2.feature,wr21_scaled,'pk2_feat1_scaled')
# static_plot_compare_labels(pk2,pk2.feature2,wr22_scaled,'pk2_feat2_scaled')
#


