from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
import pandas as pd
import shinyswatch
from data_wrangling.data_wrangling import collect_data, mood_classification, correlation, data_manipulation
from recommender_models.models import prep_for_modelling, normalise_data, recs
# from clustering.clustering_models import get_numerical_data,cluster_ready_data,run_kmeans
import seaborn as sns
from pathlib import Path
import plotly.express as px
import numpy as np
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
import tabulate
import os



# style
plt.style.use('ggplot')
page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
)

# data vars
analysis_data = data_manipulation()


    
#feature engineering
data = collect_data()
    
#feature engineering

mean_energy = analysis_data['energy'].mean()
mean_danceability = analysis_data['danceability'].mean()
mean_liveness = analysis_data['liveness'].mean()
median_instrumentalness = analysis_data['instrumentalness'].median()
mean_duration = analysis_data['duration_ms'].mean()
mean_valence = analysis_data['valence'].mean()
mean_speechiness = analysis_data['speechiness'].mean()



mood_classified = analysis_data.copy()

mood_classified['mood'] = mood_classified.apply(mood_classification, axis = 1)

grouped_by_mood = mood_classified\
                    .groupby(['track_genre','mood','track_name','artists'])\
                        .mean('popularity')\
                            .reset_index()\
                                .sort_values(by='popularity', 
                                            ascending =  False) 



my_cmap = plt.get_cmap("tab20b")

list_of_questions_mood = ['Top 10 songs where mood is happy',
                     'Top 10 songs where mood is sad',
                     'Top 10 songs where mood is energetic']

list_of_questions_genre = ['Top 10 songs where genre is pop',
                     'Top 10 songs where genre is rock',
                     'Top 10 songs where mood is hip-hop']

colours = [my_cmap(i) for i in range(len(grouped_by_mood['mood'].unique()))]

# k_clusters = run_kmeans()

# k_means_sse = kmeans_sse()

# k_means_silhoutte = silhouette_kmeans()



key_cols = ['energy', 
            'danceability',
            'liveness',
            'instrumentalness',
            'duration_ms',
            'valence',
            'speechiness']

# App-------------------------------------------------------------------
app_ui = ui.page_fluid(ui.page_navbar(
                       shinyswatch.theme.zephyr(),
                       ui.nav(" ",
                                   ui.panel_main(
                                      ui.navset_card_tab(
                                      ui.nav("Top songs by Genre",
                                                         ui.layout_sidebar(ui.sidebar(ui.input_select(
                                                                "Genre_Selection_2", 
                                                                "Select a Genre",
                                                                grouped_by_mood['track_genre'].unique().tolist(),
                                                                selected='pop',
                                                                # multiple=False
                                  
                                                          ), ui.p("""Each Genre is a combination of key music features.
                                                           In this pane, select a Genre to output the most popular songs
                                                            .
                                                            
                                                        """), open = "desktop"),
                                                                ui.row(ui.output_text("songs_headline"),
                                                                       ui.output_plot("plot_2", width = '100%',             
                                                                                      hover = True),
                                                                        # ui.input_text_area("text_input_2",
                                                                        #                    "Ask me about songs based on Genre",
                                                                        #                     # list_of_questions_genre,
                                                                        #                      width = "100%"),
                                                                        # ui.output_text_verbatim("questions_2",
                                                                                                
                                                                        #                             )
                                                                    ),
                                                                                      
                                                                width = 2,
                                                                
                                                          )),
                                                    ui.nav("Top songs by Mood",
                                                         ui.layout_sidebar(ui.sidebar(ui.input_radio_buttons(
                                                                "Mood_Selection", 
                                                                "Select a Mood",
                                                                grouped_by_mood['mood'].unique().tolist(),
                                                                selected='happy',
                                                                # multiple=False
                                  
                                                          ), ui.p("""Each Mood is an input of key music audio features.
                                                           In this pane, select a Mood to output the most popular songs
                                                            as a function of the mood.
                                                            
                                                        """),open = 'desktop'),
                                                                ui.row(ui.output_text("songs_headline_mood"),
                                                                       ui.output_plot("plot_3", width = '100%'),
                                                                        # ui.input_text_area("text_input",
                                                                        #                    "Ask me about songs based on mood",
                                                                        #                     # list_of_questions_mood,
                                                                        #                      width = "100%"),
                                                                        #         ui.output_text_verbatim("questions"
                                                                        #                             )
                                                                                                    ),
                                                                width = 2
                                                          )),
                                                           ui.nav("Content based Music Recommender System",
                                                                ui.layout_sidebar(ui.sidebar(
                                                          ui.p("""Select a mood and a song for similar recommendations
                                                            
                                                            
                                                        """),
                                                                ui.input_radio_buttons(
                                                                "mood_selector", 
                                                                "Select a mood",
                                                                grouped_by_mood['mood'].unique().tolist(),
                                                                selected = 'happy',
 
                                  
                                                          ), ui.output_ui(
                                                                "artist_selector", 
                                                          ),ui.output_ui("ui_select"),
                                                                open = 'desktop'),
                                                                ui.row(ui.output_text("headline"),
                                                                       ui.output_table("recommendations", width = '100%', 
                                                                                      ),
                                                                        
                                                                            # ui.input_text_area("text_input","Ask me about songs"),
                                                                            # ui.output_text_verbatim("questions",
                                                                            #                         placeholder=False)
                                                                       ),
                                                                width = 2
                                                          )),
                                                       
                                                    
                                                           ui.nav("Music Features Distributions", 
                                                             ui.layout_sidebar(ui.sidebar(ui.input_select(
                                                                "Genre_Selection", 
                                                                "Select a Genre",
                                                                analysis_data['track_genre'].unique().tolist(),
                                                                selected='pop',
                                                               
                                                                # multiple=False
                                                     ), ui.p("""Each Genre is a combination of key music features.
                                                           In this pane, select a Genre to study its key features.
                                                           The vertical bars on each of the charts
                                                           signify the overall mean of that music feature.
                                                            
                                                        """)
                                                 ,open = 'desktop')
                                                ,
                                                ui.row(ui.output_text("features_headline"),
                                                       ui.output_plot("plot_1",width='100%')),
                                               
                                                 ))
                                        
                                        
                                      ),
                                  
                                    
                                  height = 14
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.jpg",height="50px", style="margin:5px;" ),
                           ui.h4( "Music Analysis")
                       ),bg="#E3EBF4",
                       header = page_dependencies,fluid = True, collapsible=True
                       
    
)
)


def server(input, output, session:Session):
    
    @output
    @render.plot
    def plot_2():
    
        fig, axs = plt.subplots( figsize = (72,72), sharex = True)
        axs.barh(y = grouped_by_mood[grouped_by_mood['track_genre'] == input.Genre_Selection_2()]\
                                .sort_values(by='popularity', ascending=False)\
                                    .head(10)['track_name'],
                 width = grouped_by_mood[grouped_by_mood['track_genre'] == input.Genre_Selection_2()]\
                               .sort_values(by='popularity', ascending=False)\
                                    .head(10)['popularity'],
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        
        axs.set_xlabel("Popularity",fontsize = 8)
        # axs.set_ylabel("Songs",fontsize = 8)
        axs.tick_params(which='major', width=0.75, length=1, labelsize=8)
        plt.subplots_adjust( left = 0.8)
        
        
      
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(2,4, figsize = (48,48), sharey = True)
        axs[0,0].hist(analysis_data['danceability'][analysis_data['track_genre'] == input.Genre_Selection()],
                    
                      ec = 'black',
                      color = '#E3EBF4',
                      )
        axs[0,1].hist(analysis_data['energy'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = '#E3EBF4',
                      )
        axs[0,2].hist(analysis_data['valence'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = '#E3EBF4',
                      )
        axs[0,3].hist(analysis_data['acousticness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = '#E3EBF4',
                      )
        axs[1,0].hist(analysis_data['liveness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = '#E3EBF4',
                      )
        axs[1,1].hist(analysis_data['instrumentalness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      
                      color = '#E3EBF4',
                      )
        axs[1,2].hist(analysis_data['loudness'][analysis_data['track_genre'] == input.Genre_Selection()],
                     
                      color = '#E3EBF4',
                      ec = 'black',
                      )
        axs[1,3].hist(analysis_data['speechiness'][analysis_data['track_genre'] == input.Genre_Selection()],
                    
                      color = '#E3EBF4',
                      ec = 'black',
                      )
        
        axs[0,0].set_xlabel('danceability', fontsize = 8)
        axs[0,0].axvline(x = analysis_data['danceability'].mean(),
                         linestyle = '--',
                         )
        axs[0,0].tick_params(which='major',
                            width=0.75, 
                            length=2.5,
                            labelsize=7,
                            rotation = 90)

        axs[0,1].set_xlabel('energy',fontsize = 8)
        axs[0,1].axvline(x = analysis_data['energy'].mean(), 
                         linestyle = '--',
                         )
        axs[0,1].tick_params(which='major',
                            width=0.75,
                            length=2.5, 
                            labelsize=7,
                            rotation = 90)

        axs[0,2].set_xlabel('valence',fontsize = 8)
        axs[0,2].axvline(x = analysis_data['valence'].mean(), 
                         linestyle = '--',
                         )
        axs[0,2].tick_params(which='major', 
                             width=0.75,
                            length=2.5,
                            labelsize=7,
                            rotation = 90)

        axs[0,3].set_xlabel('acousticness',fontsize = 8)
        axs[0,3].axvline(x = analysis_data['acousticness'].mean(), 
                         linestyle = '--',
                         )
        axs[0,3].tick_params(which='major',
                            width=0.75, 
                            length=2.5,
                            labelsize=7,
                            rotation = 90)

        axs[1,0].set_xlabel('liveness',fontsize = 8)
        axs[1,0].axvline(x = analysis_data['liveness'].mean(), 
                         linestyle = '--',
                         )
        axs[1,0].tick_params(which='major', width=0.75, length=2.5, labelsize=8)

        axs[1,1].set_xlabel('instrumentalness',fontsize = 8)
        axs[1,1].axvline(x = analysis_data['instrumentalness'].mean(), 
                         linestyle = '--',
                         )
        axs[1,1].tick_params(which='major', width=0.75, 
                             length=2.5,
                             labelsize=7,
                             rotation = 90)

        axs[1,2].set_xlabel('loudness',fontsize = 8)
        axs[1,2].axvline(x = analysis_data['loudness'].mean(), 
                         linestyle = '--',
                         )
        axs[1,2].tick_params(which='major', width=0.75, 
                             length=2.5,
                            labelsize=7,
                            rotation = 90)

        axs[1,3].set_xlabel('speechiness',fontsize = 8)
        axs[1,3].axvline(x = analysis_data['speechiness'].mean(), 
                         linestyle = '--',
                         )
        axs[1,3].tick_params(which='major', width=0.75, 
                             length=2.5, 
                             labelsize=7,
                             rotation = 90)
        

        plt.subplots_adjust(wspace = 4.5)
        fig.supylabel('Total Number of Songs',fontsize = 8)
        fig.suptitle("Distributions of key music features", fontsize = 8)
        
    
    @output
    @render.plot
    def plot_3():
    
        fig, axs = plt.subplots( figsize = (72,72), sharex = True)
        axs.barh(y = grouped_by_mood[grouped_by_mood['mood'] == input.Mood_Selection()]\
                             .sort_values(by='popularity', ascending=False)\
                                .head(10)['track_name'],
                 width = grouped_by_mood[grouped_by_mood['mood'] == input.Mood_Selection()]\
                             .sort_values(by='popularity', ascending=False)\
                                .head(10)['popularity'],
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        
        # axs.set_title("Popular Tracks", fontsize = 8)
        axs.set_xlabel("Popularity",fontsize = 8)
       
        axs.tick_params(which='major', width=0.75, length=2.5, labelsize=8)
        plt.subplots_adjust( left = 0.8)
    
    @output
    @render.ui
    def artist_selector():

        data_to_model_artist = mood_classified.copy()
        data_to_model_artist = data_to_model_artist[data_to_model_artist['mood'] == input.mood_selector()]
        
        artist_list = data_to_model_artist['artists']
        # track_list = track_list.iloc[0:10000]
        artist_list = artist_list.tolist()
        return ui.input_selectize('key_artists',
                                  "Select or type your Artist",
                                  artist_list,
                               
                                  )
     

    @output
    @render.ui
    def ui_select():

        data_to_model = mood_classified.copy()
        data_to_model = data_to_model[data_to_model['mood'] == input.mood_selector()]
        data_to_model = data_to_model[data_to_model['artists'] == input.key_artists()]
        data_to_model = data_to_model.dropna()
        track_list = data_to_model['track_name']
        # track_list = track_list.iloc[0:10000]
        track_list = track_list.tolist()
        return ui.input_selectize('key_tracks',
                                  "Select or type your Song",
                                  track_list,
                                  selected = "Marie Marie"
                                  )

        
    @output
    @render.table
    def recommendations():
        top_recs = recs(recommendations_for = input.key_tracks(),
                        mood_input = input.mood_selector())
      
        
        top_recs = top_recs.style\
                   .set_properties(**{                                                  
                                    # 'background-color': 'grey',                       
                                    'border-color': 'white',
                                    'color':'black',
                                    'hide-index': True,
                                    'header-color': 'grey'})\
                                    .background_gradient(cmap = my_cmap.colors)\
                                    .hide(axis = 'index')
        
        cell_hover = {  # for row hover use <tr> instead of <td>
            'selector': 'td:hover',
            'props': [('background-color', '#E3EBF4')]
        }
        
        headers = {
            'selector': 'th:not(.index_name)',
            'props': 'background-color: #E3EBF4; font-weight:bold;color: black;'
        }

        top_recs.set_table_styles([cell_hover, headers
        ], overwrite=False)       
        
        return top_recs
    
    @output
    @render.text
    def headline():
        return f'Your top 10 recommendations for "{input.key_tracks()}" are:'
    
    @output
    @render.text
    def songs_headline():
        return f'Your top 10 "{input.Genre_Selection_2()}" songs are:'
    
    @output
    @render.text
    def songs_headline_mood():
        return f'Your top 10 "{input.Mood_Selection()}" songs are:'
    
    @output
    @render.text
    def features_headline():
        return f'Audio features of "{input.Genre_Selection()}" genre are shown below:'
    
    
    # @output
    # @render.text
    # def questions():
    #     return agent.run(input.text_input())
    
    # @output
    # @render.text
    # def questions_2():
    #     return agent.run(input.text_input_2())

www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)
