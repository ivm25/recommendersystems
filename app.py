from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
import shinyswatch
from data_wrangling.data_wrangling import collect_data, mood_classification, correlation, data_manipulation
from recommender_models.models import prep_for_modelling, normalise_data, recs
from clustering.clustering_models import get_numerical_data,cluster_ready_data,run_kmeans
import seaborn as sns
from pathlib import Path
import plotly.express as px
import numpy as np


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
colours = [my_cmap(i) for i in range(len(grouped_by_mood['mood'].unique()))]

k_clusters = run_kmeans()

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
                                      ui.nav("Top songs and artists by Genre",
                                                         ui.layout_sidebar(ui.panel_sidebar(ui.input_select(
                                                                "Genre_Selection_2", 
                                                                "Select a Genre",
                                                                grouped_by_mood['track_genre'].unique().tolist(),
                                                                selected='pop',
                                                                # multiple=False
                                  
                                                          ), ui.p("""Each Genre is a combination of key music features.
                                                           In this pane, select a Genre to output the most popular songs
                                                            and artists.
                                                            
                                                        """), width = 2),
                                                                ui.output_plot("plot_2", width = '100%'),
                                                                width = 2
                                                          )),
                                                    ui.nav("Top songs and artists by mood",
                                                         ui.layout_sidebar(ui.panel_sidebar(ui.input_radio_buttons(
                                                                "Mood_Selection", 
                                                                "Select a Mood",
                                                                grouped_by_mood['mood'].unique().tolist(),
                                                                selected='happy',
                                                                # multiple=False
                                  
                                                          ), ui.p("""Each Mood is an input of key music audio features.
                                                           In this pane, select a Mood to output the most popular songs
                                                            and artists as a function of the mood.
                                                            
                                                        """), width = 2),
                                                                ui.output_plot("plot_3", width = '100%'),
                                                                width = 2
                                                          )),
                                                           ui.nav("Recommender System",
                                                                ui.layout_sidebar(ui.panel_sidebar(
                                                          ui.p("""Select a mood and a song for similar recommendations
                                                            
                                                            
                                                        """),
                                                                ui.input_radio_buttons(
                                                                "mood_selector", 
                                                                "Select a mood",
                                                                grouped_by_mood['mood'].unique().tolist(),
                                                                selected = 'happy',
                                                                # placeholder = "Type a Song",
                                                                
                                  
                                                          ), ui.output_ui("ui_select"),width = 2),
                                                                ui.row(ui.output_text("headline"),
                                                                       ui.output_table("recommendations", width = '100%'),
                                                                       ),
                                                                width = 2
                                                          )),
                                                          ui.nav("cluster analysis",
                                                                ui.layout_sidebar(ui.panel_sidebar(
                                                          ui.p("""In this pane, select a Mood to limit the songs data matching
                                                            that mood. After that, select a song from the dropdown to
                                                            get an output of similar songs, within that mood categorisation.
                                                            
                                                            
                                                        """),
                                                                ui.input_radio_buttons(
                                                                "feature_selector", 
                                                                "Select a music feature",
                                                                key_cols,
                                                                selected = 'energy',
                                                                # placeholder = "Type a Song",
                                                                
                                  
                                                          ),width = 2),
                                                                ui.row(ui.output_plot("cls", width = '100%')
                                                                       
                                                                       ),
                                                                width = 2
                                                          )),
                                                           ui.nav("Key Music Features", 
                                                ui.layout_sidebar(ui.panel_sidebar(ui.input_select(
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
                                                 ,width = 2, )
                                                ,
                                                ui.output_plot("plot_1",width='100%'),
                                               
                                                 ))
                                        
                                        
                                      ),
                                  
                                    
                                  height = 14
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.jpg",height="50px", style="margin:5px;" ),
                           ui.h1( "Music Analysis")
                       ),bg="#E3EBF4",
                       header = page_dependencies
                       
    
)
)


def server(input, output, session:Session):
    
    @output
    @render.plot
    def plot_2():
    
        fig, axs = plt.subplots(1,2, figsize = (24,24), sharex = True)
        axs[0].barh(y = grouped_by_mood['track_name'][grouped_by_mood['track_genre'] == input.Genre_Selection_2()].iloc[0:10],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:10]),
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        axs[0].title.set_text("Most Popular Tracks")
        axs[0].set_xlabel("Popularity",fontsize = 12)
        axs[0].set_ylabel("Songs",fontsize = 12)
        # axs[0].legend(loc = "lower left")

        axs[1].barh(y = grouped_by_mood['artists'][grouped_by_mood['track_genre'] == input.Genre_Selection_2()].iloc[0:10],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:10]),
                 color = my_cmap.colors,
                 alpha = 0.7,
                 
                 )
        
        axs[1].title.set_text("Most Popular Artists")
        axs[1].set_xlabel("Popularity", fontsize = 12)
        axs[1].set_ylabel("Artists", fontsize = 12)
        
        
      
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(2,4, figsize = (48,48))
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
        
        axs[0,0].set_xlabel('danceability', fontsize = 10)
        axs[0,0].axvline(x = analysis_data['danceability'].mean(),
                         linestyle = '--',
                         )
        

        axs[0,1].set_xlabel('energy',fontsize = 10)
        axs[0,1].axvline(x = analysis_data['energy'].mean(), 
                         linestyle = '--',
                         )

        axs[0,2].set_xlabel('valence',fontsize = 10)
        axs[0,2].axvline(x = analysis_data['valence'].mean(), 
                         linestyle = '--',
                         )
        
        axs[0,3].set_xlabel('acousticness',fontsize = 10)
        axs[0,3].axvline(x = analysis_data['acousticness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,0].set_xlabel('liveness',fontsize = 10)
        axs[1,0].axvline(x = analysis_data['liveness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,1].set_xlabel('instrumentalness',fontsize = 10)
        axs[1,1].axvline(x = analysis_data['instrumentalness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,2].set_xlabel('loudness',fontsize = 10)
        axs[1,2].axvline(x = analysis_data['loudness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,3].set_xlabel('speechiness',fontsize = 10)
        axs[1,3].axvline(x = analysis_data['speechiness'].mean(), 
                         linestyle = '--',
                         )
        
        plt.subplots_adjust(wspace = 0.05)
        fig.supylabel('Total Number of Songs',fontsize = 12)
        fig.suptitle("Distributions of key music features", fontsize = 12)
        
    
    @output
    @render.plot
    def plot_3():
    
        fig, axs = plt.subplots(1,2, figsize = (24,24), sharex = True)
        axs[0].barh(y = grouped_by_mood['track_name'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:10],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:10]),
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        axs[0].title.set_text("Most Popular Tracks")
        axs[0].set_xlabel("Popularity",fontsize = 12)
        axs[0].set_ylabel("Songs",fontsize = 12)
        # axs[0].legend(loc = "lower left")

        axs[1].barh(y = grouped_by_mood['artists'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:10],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:10]),
                 color = my_cmap.colors,
                 alpha = 0.7,
                 
                 )
        
        axs[1].title.set_text("Most Popular Artists")
        axs[1].set_xlabel("Popularity", fontsize = 12)
        axs[1].set_ylabel("Artists", fontsize = 12)

    @output
    @render.ui
    def ui_select():

        data_to_model = mood_classified.copy()
        data_to_model = data_to_model[data_to_model['mood'] == input.mood_selector()]
        track_list = data_to_model['track_name']
        track_list = track_list.iloc[0:2000]
        track_list = track_list.tolist()
        return ui.input_select('key_tracks',
                                  "select you song",
                                  track_list,
                                  selected = "Lovers Rock"
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
    @render.plot
    def cls():

        fig, ax = plt.subplots(2,2,
                               sharex = True,
                               sharey = True)
        
    

        ax[0,0].scatter(k_clusters[input.feature_selector()][k_clusters['mood'] == 'energetic'],
                     k_clusters['popularity'][k_clusters['mood'] == 'energetic'],
                     
                     c = k_clusters['labels'][k_clusters['mood'] == 'energetic'],
                     label = k_clusters['labels'][k_clusters['mood'] == 'energetic']) 
        ax[0,0].title.set_text('energetic songs')
      
        
        ax[0,1].scatter(k_clusters[input.feature_selector()][k_clusters['mood'] == 'happy'],
                     k_clusters['popularity'][k_clusters['mood'] == 'happy'],
                     
                     c = k_clusters['labels'][k_clusters['mood'] == 'happy'],
                     label = k_clusters['labels'][k_clusters['mood'] == 'happy']) 
        ax[0,1].title.set_text('happy songs')
        

        ax[1,0].scatter(k_clusters[input.feature_selector()][k_clusters['mood'] == 'sad'],
                     k_clusters['popularity'][k_clusters['mood'] == 'sad'],
                     
                     c = k_clusters['labels'][k_clusters['mood'] == 'sad'],
                     label = k_clusters['labels'][k_clusters['mood'] == 'sad']) 
        ax[1,0].title.set_text('sad songs')

        ax[1,1].scatter(k_clusters[input.feature_selector()][k_clusters['mood'] == 'calm'],
                     k_clusters['popularity'][k_clusters['mood'] == 'calm'],
                     
                     c = k_clusters['labels'][k_clusters['mood'] == 'calm'],
                     label = k_clusters['labels'][k_clusters['mood'] == 'calm']) 
        ax[1,1].title.set_text('calm songs')
        
        fig.supxlabel(input.feature_selector())
        fig.supylabel('popularity')
        fig.suptitle('k-means clustering showing popularity as a function of key music features')
        plt.legend(labels = k_clusters['labels'].unique())
       

www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)
