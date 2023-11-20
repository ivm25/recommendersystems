from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
import shinyswatch
from data_wrangling.data_wrangling import collect_data, mood_classification, correlation
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

genre = collect_data()


    
#feature engineering
data = collect_data()
    
#feature engineering

mean_energy = genre['energy'].mean()
mean_danceability = genre['danceability'].mean()
mean_liveness = genre['liveness'].mean()
median_instrumentalness = genre['instrumentalness'].median()
mean_duration = genre['duration_ms'].mean()
mean_valence = genre['valence'].mean()
mean_speechiness = genre['speechiness'].mean()



mood_classified = genre.copy()

mood_classified['mood'] = mood_classified.apply(mood_classification, axis = 1)

grouped_by_mood = mood_classified\
                    .groupby(['track_genre','mood','track_name','artists'])\
                        .mean('popularity')\
                            .reset_index()\
                                .sort_values(by='popularity', 
                                            ascending =  False) 


correlation_data = correlation(genre)

colours = np.where(grouped_by_mood['popularity'] > 95, 'green','red')
my_cmap = plt.get_cmap("Dark2")


app_ui = ui.page_fluid(ui.page_navbar(
                       shinyswatch.theme.darkly(),
                       ui.nav(" ",
                                   ui.panel_main(
                                      ui.navset_card_tab(
                                          ui.nav("Key Music Features", 
                                                ui.layout_sidebar(ui.panel_sidebar(ui.input_select(
                                                                "Genre_Selection", 
                                                                "Select a Genre",
                                                                genre['track_genre'].unique().tolist(),
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
                                               
                                                 )),ui.nav("Top songs and artists",
                                                         ui.layout_sidebar(ui.panel_sidebar(ui.input_radio_buttons(
                                                                "Mood_Selection", 
                                                                "Select a Mood",
                                                                grouped_by_mood['mood'].unique().tolist(),
                                                                selected='happy',
                                                                # multiple=False
                                  
                                                          ), width = 2),
                                                                ui.output_plot("plot_2", width = '100%'),
                                                                width = 2
                                                          )),
                                                    ui.nav("Correlation Analysis",
                                                         ui.layout_sidebar(ui.panel_sidebar(
                                                           ui.p("""Looking at the Pearson Correlation Coefficient.
                                                            A positive Coefficient dhows a linear relationship between the variables
                                                                and vice versa.
                                                        """),width = 2),
                                                                ui.output_plot("plot_3", width = '100%'),
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
        axs[0].barh(y = grouped_by_mood['track_name'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:15],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:15]),
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        axs[0].title.set_text("Most Popular Tracks")
        axs[0].set_xlabel("Popularity")
        axs[0].set_ylabel("Songs")
        # axs[0].legend(loc = "lower left")

        axs[1].barh(y = grouped_by_mood['artists'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:15],
                 width = sorted(grouped_by_mood['popularity'].iloc[0:15]),
                 color = my_cmap.colors,
                 alpha = 0.7
                 )
        
        axs[1].title.set_text("Most Popular Artists")
        axs[1].set_xlabel("Popularity", fontsize = 12)
        axs[1].set_ylabel("Artists", fontsize = 12)
       
    
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(2,4, figsize = (48,48))
        axs[0,0].hist(genre['danceability'][genre['track_genre'] == input.Genre_Selection()],
                    
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,1].hist(genre['energy'][genre['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,2].hist(genre['valence'][genre['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[0,3].hist(genre['acousticness'][genre['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[1,0].hist(genre['liveness'][genre['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      color = 'green',
                      alpha = 0.8)
        axs[1,1].hist(genre['instrumentalness'][genre['track_genre'] == input.Genre_Selection()],
                      ec = 'black',
                      
                      color = 'green',
                      alpha = 0.8)
        axs[1,2].hist(genre['loudness'][genre['track_genre'] == input.Genre_Selection()],
                     
                      color = 'green',
                      ec = 'black',
                      alpha = 0.8)
        axs[1,3].hist(genre['speechiness'][genre['track_genre'] == input.Genre_Selection()],
                    
                      color = 'green',
                      ec = 'black',
                      alpha  = 0.8)
        
        axs[0,0].set_xlabel('danceability', fontsize = 10)
        axs[0,0].axvline(x = genre['danceability'].mean(),
                         linestyle = '--',
                         )
        

        axs[0,1].set_xlabel('energy',fontsize = 10)
        axs[0,1].axvline(x = genre['energy'].mean(), 
                         linestyle = '--',
                         )

        axs[0,2].set_xlabel('valence',fontsize = 10)
        axs[0,2].axvline(x = genre['valence'].mean(), 
                         linestyle = '--',
                         )
        
        axs[0,3].set_xlabel('acousticness',fontsize = 10)
        axs[0,3].axvline(x = genre['acousticness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,0].set_xlabel('liveness',fontsize = 10)
        axs[1,0].axvline(x = genre['liveness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,1].set_xlabel('instrumentalness',fontsize = 10)
        axs[1,1].axvline(x = genre['instrumentalness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,2].set_xlabel('loudness',fontsize = 10)
        axs[1,2].axvline(x = genre['loudness'].mean(), 
                         linestyle = '--',
                         )
        
        axs[1,3].set_xlabel('speechiness',fontsize = 10)
        axs[1,3].axvline(x = genre['speechiness'].mean(), 
                         linestyle = '--',
                         )
        
        plt.subplots_adjust(wspace = 0.05)
        fig.supylabel('Total Number of Songs',fontsize = 12)
        fig.suptitle("Distributions of key music features", fontsize = 12)
        
    
    @output
    @render.plot
    def plot_3():
    
        heatmap = sns.heatmap(correlation_data, 
                              annot=True, 
                              cmap="Greens",
                                fmt='.1g')

www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)
