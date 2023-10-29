import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def collect_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    genre_data = pd.read_csv('data/dataset.csv')
    return genre_data

data = collect_data()
    
#feature engineering

mean_energy = data['energy'].mean()
mean_danceability = data['danceability'].mean()
mean_liveness = data['liveness'].mean()
median_instrumentalness = data['instrumentalness'].median()
mean_duration = data['duration_ms'].mean()
mean_valence = data['valence'].mean()
mean_speechiness = data['speechiness'].mean()




def mood_classification(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if (df['energy'] > mean_energy) & (df['danceability'] > mean_danceability):
        return 'energetic'
    elif (df['instrumentalness'] > median_instrumentalness) &  (df['duration_ms'] >  mean_duration):
        return 'calm'
    elif (df['valence'] > mean_valence) & (df['energy'] >  mean_energy):
        return 'happy'
    elif (df['liveness'] > mean_liveness):
        return 'live'
    elif (df['speechiness'] > mean_speechiness):
        return 'speechy'
    else:
        return 'sad'
    

    


if __name__ == "__main__":
    
    mood_classified = data.copy()
    
    mood_classified['mood'] = mood_classified.apply(mood_classification, axis = 1)
    
  
