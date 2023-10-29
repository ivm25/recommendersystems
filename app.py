from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
from data_wrangling.data_wrangling import collect_data, mood_classification
import seaborn as sns
from pathlib import Path
import plotly.express as px


# style
plt.style.use('ggplot')
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
                    .groupby(['track_genre','mood','track_name'])\
                        .mean('popularity')\
                            .reset_index()\
                                .sort_values(by='popularity', 
                                            ascending =  False) 

app_ui = ui.page_fluid(ui.page_navbar(
                       ui.nav(" ",
                              ui.layout_sidebar(
                                  sidebar = ui.panel_sidebar(
                                      ui.h2("Let's Analyse Music"),
                                      ui.input_selectize(
                                        "Genre_Selection", 
                                        "Select a Genre",
                                        genre['track_genre'].unique().tolist(),
                                        selected='rock',
                                        multiple=False
                                    ), ui.input_selectize(
                                        "Mood_Selection", 
                                        "Select a Mood",
                                        grouped_by_mood['mood'].unique().tolist(),
                                        selected='happy',
                                        multiple=False
                                  
                                  )),
                                  main = ui.panel_main(
                                      ui.navset_pill(
                                          ui.nav("Key Music Features",
                                                ui.row(ui.output_plot("plot_1",width='100%'))
                                                
                                                 ),ui.nav("Top Songs",
                                                         ui.row(ui.output_plot("plot_2", width = '100%'))
                                                          )
                                        
                                      ),
                                    #   ui.row(ui.output_plot("plot_1",width='100%')),
                                    
                                  )
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.jpg",height="50px", style="margin:5px;" ),
                           ui.h1( "Music Features Analyser")
                       ),bg="#0062cc",
                       header = page_dependencies
                       
    
)
)


def server(input, output, session:Session):
    
    @output
    @render.plot
    def plot_2():
        # fig = px.bar(grouped_by_mood,
        #              x = grouped_by_mood['popularity'].iloc[0:15],
        #              y = grouped_by_mood['track_name'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:15],
        #              color = 'track_genre')
        
        fig, axs = plt.subplots(figsize = (36,24))
        axs.barh(y = grouped_by_mood['track_name'][grouped_by_mood['mood'] == input.Mood_Selection()].iloc[0:15],
                 width = grouped_by_mood['popularity'].iloc[0:15],
                 color = "#0062cc",
                 
                 )
        
       
       
    
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(4,2, figsize = (36,24))
        axs[0,0].hist(genre['danceability'][genre['track_genre'] == input.Genre_Selection()],
                      
                      color = 'green')
        axs[0,1].hist(genre['energy'][genre['track_genre'] == input.Genre_Selection()],
                      
                      color = 'green')
        axs[1,0].hist(genre['valence'][genre['track_genre'] == input.Genre_Selection()],
                      
                      color = 'blue')
        axs[1,1].hist(genre['acousticness'][genre['track_genre'] == input.Genre_Selection()],
                      
                      color = 'blue')
        axs[2,0].hist(genre['liveness'][genre['track_genre'] == input.Genre_Selection()],
                     
                      color = 'orange')
        axs[2,1].hist(genre['instrumentalness'][genre['track_genre'] == input.Genre_Selection()],
                      
                      color = 'orange')
        axs[3,0].hist(genre['loudness'][genre['track_genre'] == input.Genre_Selection()],
                     
                      color = 'red')
        axs[3,1].hist(genre['speechiness'][genre['track_genre'] == input.Genre_Selection()],
                    
                      color = 'red')
        
        axs[0,0].set_xlabel('danceability', fontsize = 10)
        axs[0,1].set_xlabel('energy',fontsize = 10)
        axs[1,0].set_xlabel('valence',fontsize = 10)
        axs[1,1].set_xlabel('acousticness',fontsize = 10)
        axs[2,0].set_xlabel('liveness',fontsize = 10)
        axs[2,1].set_xlabel('instrumentalness',fontsize = 10)
        axs[3,0].set_xlabel('loudness',fontsize = 10)
        axs[3,1].set_xlabel('speechiness',fontsize = 10)
        fig.supylabel('Total Number of Songs',fontsize = 12)
        fig.suptitle("Distributions of key music features", fontsize = 12)
        plt.tight_layout()
    


www_dir = Path(__file__).parent /"www"
app = App(app_ui, server,static_assets=www_dir)