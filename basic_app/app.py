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
                                        genre['genre'].unique().tolist(),
                                        selected='Pop',
                                        multiple=False
                                    ), ui.input_selectize(
                                        "Song_Selection", 
                                        "Select a Song",
                                        genre['song_name'].unique().tolist(),
                                        selected= 'Pathology',
                                        multiple=False
                                    )
                                  
                                  ),
                                  main = ui.panel_main(
                                      ui.navset_pill(
                                          ui.nav("Distributions",
                                                
                                                
                                                 ),ui.nav("Song Categories",
                                                          )
                                        
                                      ),ui.row(ui.output_plot("plot_1",width='50%'),
                                               ui.output_plot("plot_2", width = '50%')),
                                       ui.row(ui.output_plot("plot_3",width='50%'),
                                               ui.output_plot("plot_4", width = '50%'))
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
    @render.plot
    def plot_1():
        return sns.displot(genre, 
                           y = genre['danceability'][genre['genre'] == input.Genre_Selection()],
                           
                           kde = True,
                           
                            )
    @output
    @render.plot
    def plot_2():
        return sns.histplot(genre, 
                            y =genre['energy'][genre['genre'] == input.Genre_Selection()],
                            kde=True,
                            )
                

    @output
    @render.plot
    def plot_3():
        return sns.histplot(genre,
                            y =genre['valence'][genre['genre'] == input.Genre_Selection()],
                           kde = True
                            )
    @output
    @render.plot
    def plot_4():
        return sns.histplot(genre,
                            y = genre['duration_ms'][genre['genre'] == input.Genre_Selection()],
                            kde=True,
                            )

parent_path = str(Path().absolute()) + '/basic_app/www'
www_dir = Path(__file__).parent /"www"
app = App(app_ui, server, static_assets=www_dir)
