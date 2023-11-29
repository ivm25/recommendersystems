import pandas as pd
import numpy as np
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import euclidean_distances
from scipy.spatial import distance
from scipy.spatial.distance import cdist, cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from data_wrangling.data_wrangling import collect_data, data_manipulation
from data_wrangling.data_wrangling import mood_classification


# fetch data

analysis_data = data_manipulation()

mood_classified = analysis_data.copy()

mood_classified['mood'] = mood_classified.apply(mood_classification,
                                                 axis = 1)


def prep_for_modelling():

    """_summary_

    Returns:
        _type_: _description_
    """
    key_columns = mood_classified[[
                        'danceability',
                        'energy',
                        "valence", 
                        "instrumentalness",
                        "acousticness",
                        "liveness",
                        "track_genre",
                        "loudness",
                        "speechiness",
                        'tempo',
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

# data_for_modelling = prep_for_modelling()


def normalise_data(key_col = None):
    """_summary_

    Returns:
        _type_: _description_
    """
    
    data_for_modelling = prep_for_modelling()
    data_for_modelling.index = analysis_data[key_col]
    # normalized  data by columns
    normalised_data = pd.DataFrame(normalize(data_for_modelling, axis=1))
    normalised_data.columns = data_for_modelling.columns

    normalised_data.index = data_for_modelling.index
    
    return normalised_data



def recs(recommendations_for = None,
        join_col = 'track_name',
        ):
    """_summary_

    Args:
        song (_type_): _description_

    Returns:
        _type_: _description_
    """

    df_normalised = normalise_data(key_col = 'track_name')
    all_songs = pd.DataFrame(df_normalised.index)

    recommendation_matrix = all_songs.copy()    
    recommendation_matrix['similarity'] = all_songs['track_name']\
                                             .apply(lambda x:cosine(df_normalised.loc[recommendations_for], 
                                                                     df_normalised.loc[x]))
   
    recommendation_matrix = recommendation_matrix.sort_values(by='similarity', 
                                        ) 
    # print(recommendation_matrix)                                    
    # join with the rest of columns
    recommendations = recommendation_matrix.merge(mood_classified,
                                                    how = 'left',
                                                    left_on = join_col,
                                                    right_on = join_col
                                                    )
    
    top_recs = recommendations[['track_name', 
                                'artists', 
                                'track_genre',
                                'album_name',
                                'mood']].iloc[1:10]
    # print(top_recs)
    return top_recs

