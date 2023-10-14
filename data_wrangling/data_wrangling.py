import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collect_data():
    genre_data = pd.read_csv('data/genres_v2.csv')
    return genre_data

data = collect_data()

data.info()

data['song_name'].unique()
#feature engineering

mean_energy = data['energy'].mean()
mean_danceability = data['danceability'].mean()
mean_liveness = data['liveness'].mean()
median_instrumentalness = data['instrumentalness'].median()
mean_duration = data['duration_ms'].mean()
mean_valence = data['valence'].mean()
mean_speechiness = data['speechiness'].mean()

def moods(df):
    df = data
    df['energetic'] = np.where(
                              (df['danceability'] > mean_danceability)
                               & (df['energy'] >  mean_energy),'1','')
    df['calm'] = np.where(
                         (df['instrumentalness'] > median_instrumentalness) 
                         & (df['duration_ms'] >  mean_duration),'2','')
    df['happy'] = np.where(
                         (df['valence'] > mean_valence) 
                         & (df['energy'] >  mean_energy),'3','')
    df['sad'] = np.where(
                         (df['valence'] < mean_valence) 
                         | (df['energy'] <  mean_energy),'4','')
    df['live'] = np.where(
                         (df['liveness'] > mean_liveness) 
                         ,'5','')
    df['speechy'] = np.where(
                         (df['speechiness'] > mean_liveness) 
                         ,'6','')
    
    return df

mood_data = moods(data)

def mood_classification(df):
    df = mood_data
    mood_data_melted = pd.melt(df,
                               id_vars = ['song_name',
                                            'genre'],
                               value_vars=['energetic',
                                            'happy',
                                            'sad',
                                            'calm',
                                            'live',
                                            'speechy'],
                               var_name='moods'
                                      )
    return mood_data_melted

moods_of_songs = mood_classification(mood_data)
# mood_data['mood'] = mood_data.apply(lambda x:mood_classification(mood_data), axis =1)
mood_data

