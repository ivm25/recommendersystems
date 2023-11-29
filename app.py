from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
import shinyswatch
from data_wrangling.data_wrangling import collect_data, mood_classification, correlation, data_manipulation
from recommender_models.models import prep_for_modelling, normalise_data, recs
import seaborn as sns
from pathlib import Path
import plotly.express as px
import numpy as np


# style
plt.style.use('dark_background')
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


prep_for_modelling()

normalise_data(key_col = 'track_name')



# colours = ['g' if x == 'happy' else 'b' for x in mood_classified['mood'].unique()]
my_cmap = plt.get_cmap("Dark2")
colours = [my_cmap(i) for i in range(len(grouped_by_mood['mood'].unique()))]


app_ui = ui.page_fluid(ui.page_navbar(
                       shinyswatch.theme.darkly(),
                       ui.nav(" ",
                                   ui.panel_main(
                                      ui.navset_card_tab(
                                          ui.nav("Key Music Features", 
                                                ui.layout_sidebar(ui.panel_sidebar(ui.input_select(
                                                                "Genre_Selection", 
                                                                "Select a Genre",
                                                                analysis_data['track_genre'].unique().tolist(),
                                                                selected='rock',
                                                               
                                                                # multiple=False
                                                ), ui.p("""Each Genre is a combination of key music features.
                                                           In this pane, select a Genre to study its key features.
                                                           The white vertical bars on each of the charts
                                                           signify the overall mean of that music feature.
                                                            
                                                        """)
                                                 ,width = 2, )
                                                ,
                                                ui.output_plot("plot_1",width='100%'),
                                               
                                                 )),ui.nav("Top songs and artists by Genre",
                                                         ui.layout_sidebar(ui.panel_sidebar(ui.input_select(
                                                                "Genre_Selection_2", 
                                                                "Select a Genre",
                                                                grouped_by_mood['track_genre'].unique().tolist(),
                                                                selected='rock',
                                                                # multiple=False
                                  
                                                          ), width = 2),
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
                                  
                                                          ), width = 2),
                                                                ui.output_plot("plot_3", width = '100%'),
                                                                width = 2
                                                          )),
                                                           ui.nav("Recommender System",
                                                         ui.layout_sidebar(ui.panel_sidebar(ui.input_text(
                                                                "song_recommender", 
                                                                "Song Input",
                                                                value = 'Hero',
                                                                placeholder = "Type a Song",
                                                                
                                  
                                                          ), width = 2),
                                                                ui.output_table("recommendations", width = '100%'),
                                                                width = 2
                                                          ))
                                        
                                        
                                      ),
                                  
                                    
                                  height = 14
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.jpg",height="50px", style="margin:5px;" ),
                           ui.h1( "Music Analysis")
                       ),bg="#0B9C31",
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
        
        
        # axs[1].legend(grouped_by_mood['mood'].unique(),
        #               loc = 'best',
        #               handlelength=4,
        #               borderpad=2,
        #               fontsize = 8,
        #            prop = {'size':10})
    
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(2,4, figsize = (48,48))
        axs[0,0].hist(analysis_data['danceability'][analysis_data['track_genre'] == input.Genre_Selection()],
                    
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,1].hist(analysis_data['energy'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,2].hist(analysis_data['valence'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,3].hist(analysis_data['acousticness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[1,0].hist(analysis_data['liveness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[1,1].hist(analysis_data['instrumentalness'][analysis_data['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      
                      color = 'green',
                      alpha = 0.8)
        axs[1,2].hist(analysis_data['loudness'][analysis_data['track_genre'] == input.Genre_Selection()],
                     
                      color = 'green',
                      ec = 'black',
                      alpha = 0.8)
        axs[1,3].hist(analysis_data['speechiness'][analysis_data['track_genre'] == input.Genre_Selection()],
                    
                      color = 'green',
                      ec = 'black',
                      alpha  = 0.8)
        
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
    @render.table
    def recommendations():
        top_recs = recs(recommendations_for=input.song_recommender())
      
        
        top_recs = top_recs.style\
                   .set_properties(**{                                                  
                                    # 'color': '#0B9C31',                       
                                    'border-color': 'white',
                                    'hide-index': True,
                                    'header-color': 'black'})
                                    
        
        return top_recs
    
www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)
