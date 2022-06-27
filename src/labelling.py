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

def find_similar_material(spettrofeat, database, spettro='row1col1', printa=True):
    '''' dato un spettro di feature di un campione di spettri ed un database'''
    dist_dic = dict()
    for key in database.index:
        dist_dic[key] = euclidean(spettrofeat.loc[spettro, :], database.loc[key, :])
    min = np.min(list(dist_dic.values()))
    index = list(dist_dic.values()).index(min)
    if printa:
        print(f'+ simile = {list(dist_dic.keys())[index]} con {min}')
    return dist_dic, list(dist_dic.keys())[index], min


def wrap(spettrofeat, database):
    ''' wrapper di find_similar_material con tutto un database di spettri'''
    labels = []
    scores = []
    for key in spettrofeat.index:
        _j, label, score = find_similar_material(spettrofeat, database, spettro=key, printa=False)
        scores.append(score)
        labels.append(label)

    return np.array(labels), np.array(scores)


def weighthed_scale(df_to_transf,ranges, second_df_to_transf = None, save = False, **columns_bunch):
    transformers = []
    bunch_names = list(columns_bunch.keys())
    cols_flattened = []
    if len(ranges) != len(columns_bunch):
        print('PROBLEMAPROBLEMAPROBLEMA')
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





    # scalerdb_mean = MinMaxScaler(feature_range=(0, 400))
    # scalerpk1_mean = MinMaxScaler(feature_range=(0, 400))
    # scalerdb_std = MinMaxScaler(feature_range=(0, 150))
    # scalerpk1_std = MinMaxScaler(feature_range=(0, 150))
    # scalerdb_K = MinMaxScaler(feature_range=(0, 600))
    # scalerpk1_K = MinMaxScaler(feature_range=(0, 600))
    # # scalo soloo le colonne non inerenti al numero d'onda, perchè mi va bene che ad esso venga data più importanza
    # col_to_scale_mean = [f'{prop}_mean' for prop in ['prominences', 'peak_heights']]
    # col_to_scale_std = [f'{prop}_std' for prop in ['prominences', 'peak_heights']]
    # col_to_scale_K = ['K_std']
    # coltranfpk = ColumnTransformer(transformers=[('scaled_means', scalerpk1_mean, col_to_scale_mean),
    #                                              ('scaled_stds', scalerpk1_std, col_to_scale_std),
    #                                              ('scaled_K', scalerpk1_K, col_to_scale_K)])
    # coltranfdb = ColumnTransformer(transformers=[('scaled_means', scalerdb_mean, col_to_scale_mean),
    #                                              ('scaled_std', scalerdb_std, col_to_scale_std),
    #                                              ('scaled_K', scalerdb_K, col_to_scale_K)])
    # # Ricreo il dataframe con le colonne non scalate per prime
    # scaled_database_feat1 = pd.DataFrame(database_feat1, columns=['K_mean'], index=database_feat1.index)
    # scaled_pk1_feat1 = pd.DataFrame(pk1.feature, columns=['K_mean'], index=pk1.feature.index)
    # # AGGIUNGO  le colonne riscalate
    # scaled_database_feat1[col_to_scale_K + col_to_scale_mean + col_to_scale_std] = coltranfdb.fit_transform(
    #     database_feat1)
    # scaled_pk1_feat1[col_to_scale_K + col_to_scale_mean + col_to_scale_std] = coltranfpk.fit_transform(pk1.feature)

#
pk2 = Spettri(datas.data2)
pk1 = Spettri(datas.data1)
pk2.peakfinder()
pk1.peakfinder()
pk2.featextract(statlist=['mean','std','50%'])
pk1.featextract(statlist=['mean','std','50%'])
pk1.featextract2()
pk2.featextract2()
database_picchi = funzioni.peakfinder(datas.database, prop = pk1.prop)
database_feat1 = funzioni.featextract1_df(database_picchi,statlist=['mean','std','50%'])
database_feat2 = funzioni.featextract2_df(database_picchi)

scaled_pk1_feat1 = weighthed_scale(pk1.feature,[(100,1000),(0,600),(0,400)],K_mean = ['K_mean'], p_h = ['prominences_mean','peak_heights_mean','K_50%'], std_s= ['prominences_std','K_std','peak_heights_std'])
scaled_database_feat1 = weighthed_scale(database_feat1,[(100,1000),(0,600),(0,400)],K_mean = ['K_mean'], p_h = ['prominences_mean','peak_heights_mean','K_50%'], std_s= ['prominences_std','K_std','peak_heights_std'])

scaled_pk1_feat2 = weighthed_scale(pk1.feature2,[(100,1000),(0,600),(0,400),(0,200)],pk_K = [f'pk_{n}_K' for n in range(1,11)], pk_hei = [f'pk_{n}_peak_heights' for n in range(1,11)], pk_pro= [f'pk_{n}_prominences' for n in range(1,11)], pk_widths = [f'pk_{n}_widths' for n in range(1,11)] )
scaled_database_feat2 = weighthed_scale(database_feat2,[(100,1000),(0,600),(0,400),(0,200)],pk_K = [f'pk_{n}_K' for n in range(1,11)], pk_hei = [f'pk_{n}_peak_heights' for n in range(1,11)], pk_pro= [f'pk_{n}_prominences' for n in range(1,11)], pk_widths = [f'pk_{n}_widths' for n in range(1,11)])

wr11,score11 = wrap(pk1.feature,database_feat1)
wr12,score12 = wrap(pk1.feature2,database_feat2)

wr21,score21 = wrap(pk2.feature,database_feat1)
wr22, score22 = wrap(pk2.feature2,database_feat2)

wr11_scaled, score11_scaled = wrap(scaled_pk1_feat1, scaled_database_feat1)
wr12_scaled,score12_scaled = wrap(scaled_pk1_feat2, scaled_database_feat2)