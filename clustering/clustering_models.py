import pandas as pd
import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import euclidean_distances
from sklearn.metrics import silhouette_score

from scipy.spatial import distance
from scipy.spatial.distance import cdist, cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from data_wrangling.data_wrangling import collect_data, data_manipulation, mood_classification

import matplotlib.pyplot as plt
import seaborn as sns

# fetch data

plt.style.use('ggplot')

analysis_data = data_manipulation()

mood_classified = analysis_data.copy()

mood_classified['mood'] = mood_classified.apply(mood_classification,
                                                 axis = 1)


def get_numerical_data():

    """_summary_

    Returns:
        _type_: _description_
    """
    data_to_model = analysis_data.copy()
    

    key_columns = data_to_model[[
                        'danceability',
                        'energy',
                        "valence", 
                        "instrumentalness",
                        "acousticness",
                        "liveness",
                        "track_genre",
                        "loudness",
                        "speechiness",
                        # 'tempo',
                        'duration_ms',
                        # 'popularity'
                        # 'time_signature'
                        # 'mode'
                        ]]

    df = key_columns.copy()

    dummy_data = pd.get_dummies(df, 
                                prefix = 'genre', 
                                columns = ['track_genre'],
                                dtype = 'int')

    return dummy_data


def cluster_ready_data(
                   key_col = None,
                   ):
    """_summary_

    Returns:
        _type_: _description_
    """
    data_to_model = analysis_data.copy()
   

    data_for_modelling = get_numerical_data()
    data_for_modelling.index = data_to_model[key_col]
    # normalized  data by columns
    normalised_data = pd.DataFrame(normalize(data_for_modelling, axis=1))
    normalised_data.columns = data_for_modelling.columns

    normalised_data.index = data_for_modelling.index
    
    return normalised_data


def run_kmeans():

    clustered_data = cluster_ready_data(key_col = 'track_name')

    kmeans = KMeans(init="k-means++",
                    n_clusters=4,
                    random_state=15).fit(clustered_data)


    mood_classified['labels'] = kmeans.labels_
    labelled_data = mood_classified

    return labelled_data


test = run_kmeans()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')



# ax.scatter(mood_classified['liveness'],
#              mood_classified['speechiness'],
#              mood_classified['loudness'],
#              c = mood_classified['labels'],
#              label = mood_classified['mood']) 

# ax.set_xlabel('energy')
# ax.set_ylabel('danceability')
# ax.set_zlabel('loudness')
# plt.show()

#
# plt.style.use('ggplot')
# sns.catplot(
#     data=test, 
#     x="loudness",
#     y="popularity", hue="labels",
#     native_scale=True, zorder=1,
#     # col = 'mood'
# )

# plt.show()

 # A list holds the SSE values for each k

def kmeans_sse():
    sse = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, init="k-means++",
                        random_state=15)
        kmeans.fit(cluster_ready_data(key_col = 'track_name'))
        sse.append(kmeans.inertia_)
    return sse


# plt.style.use("ggplot")
# plt.plot(range(1, 20), sse)
# plt.xticks(range(1, 20))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()

#------------

def silhouette_kmeans():

    silhouette_coefficients = []
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, init="k-means++",
                        random_state=15)
        kmeans.fit(cluster_ready_data(key_col = 'track_name'))
        score = silhouette_score(cluster_ready_data(key_col = 'track_name'), kmeans.labels_)
        silhouette_coefficients.append(score)
    return silhouette_coefficients


# plt.plot(range(2, 11), silhouette_coefficients)
# plt.xticks(range(2, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# plt.show()