from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
from data_wrangling.data_wrangling import collect_data, mood_classification
import seaborn as sns
from pathlib import Path

www_dir = Path(__file__).parent / "www"

# style
sns.set_style('dark')
page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css")
)

# data vars

genre = collect_data()

app_ui = ui.page_navbar(
                       ui.nav(" ",
                              ui.layout_sidebar(
                                  sidebar = ui.panel_sidebar(
                                      ui.h2("Let's Analyse Music"),
                                      ui.input_selectize(
                                        "Genre_Selection", 
                                        "Select a Genre",
                                        genre['track_genre'].unique().tolist(),
                                        selected='Pop',
                                        multiple=False
                                    ), 
                                  
                                  ),
                                  main = ui.panel_main(
                                      ui.navset_pill(
                                          ui.nav("Distributions",
                                                 ui.row(ui.output_plot("plot_1",width='100%')),
                                                
                                                 ),ui.nav("Song Categories",
                                                          ui.output_text_verbatim("commentary")
                                                          )
                                        
                                      ),
                                            #    ui.output_plot("plot_2", width = '50%')),
                                    #    ui.row(ui.output_plot("plot_3",width='50%'),
                                    #            ui.output_plot("plot_4", width = '50%'))
                                  )
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.jpg",height="50px", style="margin:5px;" ),
                           ui.h1( "Music Features Analyser")
                       ),bg="#0062cc",
                       header = page_dependencies
                       
    
)



def server(input, output, session:Session):
    @output
    @render.text
    def commentary():
        return "This is under construction"
    
    @output
    @render.plot
    def plot_1():
        fig, axs = plt.subplots(4,2)
        axs[0,0].hist(genre['danceability'][genre['track_genre'] == input.Genre_Selection()],
                      histtype='step',
                      color = 'green')
        axs[0,1].hist(genre['energy'][genre['track_genre'] == input.Genre_Selection()])
        axs[1,0].hist(genre['valence'][genre['track_genre'] == input.Genre_Selection()])
        axs[1,1].hist(genre['acousticness'][genre['track_genre'] == input.Genre_Selection()])
        axs[2,0].hist(genre['liveness'][genre['track_genre'] == input.Genre_Selection()])
        axs[2,1].hist(genre['instrumentalness'][genre['track_genre'] == input.Genre_Selection()])
        axs[3,0].hist(genre['loudness'][genre['track_genre'] == input.Genre_Selection()])
        axs[3,1].hist(genre['speechiness'][genre['track_genre'] == input.Genre_Selection()])
        
        axs[0,0].set_xlabel('danceability')
        axs[0,1].set_xlabel('energy')
        axs[1,0].set_xlabel('valence')
        axs[1,1].set_xlabel('acousticness')
        axs[2,0].set_xlabel('liveness')
        axs[2,1].set_xlabel('instrumentalness')
        axs[3,0].set_xlabel('loudness')
        axs[3,1].set_xlabel('speechiness')
        plt.tight_layout()
                        #   sns.displot(genre, 
                        #    y = genre['danceability'][genre['genre'] == input.Genre_Selection()],
                           
                        #    kde = True,
                           
                        #     )
    # @output
    # @render.plot
    # def plot_2():
    #     return sns.histplot(genre, 
    #                         y =genre['energy'][genre['genre'] == input.Genre_Selection()],
    #                         kde=True,
    #                         )
                

    # @output
    # @render.plot
    # def plot_3():
    #     return sns.histplot(genre,
    #                         y =genre['valence'][genre['genre'] == input.Genre_Selection()],
    #                        kde = True
    #                         )
    # @output
    # @render.plot
    # def plot_4():
    #     return sns.histplot(genre,
    #                         y = genre['duration_ms'][genre['genre'] == input.Genre_Selection()],
    #                         kde=True,
    #                         )

parent_path = str(Path().absolute()) + '/basic_app/www'
www_dir = Path(__file__).parent /"www"
app = App(app_ui, server, static_assets=www_dir)
