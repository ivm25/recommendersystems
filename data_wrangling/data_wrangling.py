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

def mood_classification(df):
    df = data
    df['energetic'] = np.where(
                              (df['danceability'] > mean_danceability)
                               & (df['energy'] >  mean_energy),'energetic','')
    df['calm'] = np.where(
                         (df['instrumentalness'] > median_instrumentalness) 
                         & (df['duration_ms'] >  mean_duration),'calm','')
    df['happy'] = np.where(
                         (df['valence'] > mean_valence) 
                         & (df['energy'] >  mean_energy),'happy','')
    df['sad'] = np.where(
                         (df['valence'] < mean_valence) 
                         | (df['energy'] <  mean_energy),'sad','')
    df['live'] = np.where(
                         (df['liveness'] > mean_liveness) 
                         ,'live','')
    df['speechy'] = np.where(
                         (df['speechiness'] > mean_liveness) 
                         ,'speechy','')
    
    return df

mood_data = mood_classification(data)

