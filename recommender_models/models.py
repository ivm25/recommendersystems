import pandas as pd
import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import euclidean_distances
from scipy.spatial import distance
from scipy.spatial.distance import cdist, cosine, euclidian
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from data_wrangling.data_wrangling import collect_data
from data_wrangling.data_wrangling import mood_classification

analysis_data = collect_data()

mood_classified = analysis_data.copy()

mood_classified['mood'] = mood_classified.apply(mood_classification,
                                                 axis = 1)

# analysis_data['track_id'] = analysis_data.index

df = mood_classified[[
                    'danceability',
                    'energy',
                    "valence", 
                    "instrumentalness",
                    "acousticness",
                    "liveness",
                    "loudness",
                    "speechiness",
                    "popularity",
                    ]]




df.index = analysis_data['track_id']
# normalized  data by columns
df_normalized = pd.DataFrame(normalize(df, axis=1))
df_normalized.columns = df.columns

df_normalized.index = df.index
all_songs = pd.DataFrame(df_normalized.index)

def recs(id):


   

    all_songs['distance'] = all_songs['track_id'].apply(lambda x: cosine(df_normalized.loc[id], df_normalized.loc[x]))

    # top_songs = all_songs.sort_values(['distance']).head(5)

    return all_songs

# recommended_songs = recs('2hETkH7cOfqmz3LqZDHZf5')
# similarity = cosine_similarity(df_normalized[0:2000])

# track_id = "2C3TZjDRiAzdyViavDJ217"

# similar_songs = similarity[1].argsort()[:-11:-1]

def recs_2(id):
    dis = []
    for i,z in enumerate(all_songs['track_id'].unique()):
        # print(i)
        sim = cosine(df_normalized.loc[id], df_normalized.iloc[i])
        dis.append(sim)
  
    return all_songs

