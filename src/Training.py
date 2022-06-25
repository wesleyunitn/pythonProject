from src.classes import Spettri,plot_spettri,plot_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_load import *
from src import cluster_routine
import funzioni
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score,davies_bouldin_score
from sklearn.decomposition import PCA

