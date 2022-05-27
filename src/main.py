from src.classes import Spettri
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_load import *
from src import cluster_routine

#CREO DUE ISTANZE DI PICCHI DAGLI STESSI DATI CON CONDIZIONI DIVERSE
# picchi fissati
plt.subplot
picchi1 = Spettri(data1, npicchi = None,prop={'height':10^-5,'prominence':10^-5})

# picchi1.normalizer()
picchi1.peakfinder()
# plt.xlabel('N_clusters')
# plt.title('campione1')
# cluster_routine.km_cluster_plt(picchi1.featextract(cols=['peak_heights','K','prominences'],statlist=['mean','std','count']),pca=2)

picchi1.feat1 = picchi1.featextract(cols=['prominences','K','peak_heights'],statlist=['mean','std','count'])

kmeanslist1 = cluster_routine.km_cluster_plt(picchi1.feat1)
picchi1.feat1['labels_dbscan'] = cluster_routine.db_cluster_plt(picchi1.feat1, n_components=3).labels_
picchi1.feat1['labels_km']= kmeanslist1[5].labels_
sns.pairplot(data=picchi1.feat1,vars=['prominences_mean','count','K_mean','peak_heights_mean'],hue='labels_dbscan')
plt.show()






# print(np.unique(feat.set_index(['labels']).index,return_counts=True))
# picchi2 = Spettri(data2, npicchi = 40,)
# picchi2.normalizer()
# picchi2.peakfinder()
# print(picchi2.picchi[0])
# kmeanslist2 = cluster_routine.km_cluster_plt(picchi2.featextract(cols=['prominences','K'],))



