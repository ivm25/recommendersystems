from shiny import App, render, ui, reactive, Session
from matplotlib import pyplot as plt
from data_wrangling.data_wrangling import collect_data
import seaborn as sns
from pathlib import Path

# style
sns.set_style('darkgrid')
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
                                        ['Rock', 'Pop'],
                                        selected='Pop',
                                        multiple=False
                                    )
                                  
                                  ),
                                  main = ui.panel_main(
                                      ui.navset_pill_card(
                                          ui.nav("Distributions"
                                                
                                                 ),
                                        
                                      ),ui.row(ui.output_plot("test_plot",width='50%'),
                                               ui.output_plot("plot_2", width = '50%'))
                                  )
                              ),
                       ),
                       title=ui.tags.div(
                           ui.img(src = "ishan_logo.png"),
                           ui.h1( "Music Features Analyser")
                       ),bg="#0062cc"
                       
    
)

# app_ui = ui.page_fluid(
#     ui.h2("Hello world of Rec Systems"),
#     ui.input_slider("n", "N", 0, 140, 20),
#     ui.output_text("txt"),
#     ui.output_plot("test_plot"),
# )


def server(input, output, session:Session):
    # @output
    # @render.text
    # def txt():
    #     return f"n*2 is {input.n() * 2}"
    @output
    @render.plot
    def test_plot():
        return sns.histplot(genre['instrumentalness'],
                            kde=True,
                            )
    @output
    @render.plot
    def plot_2():
        return sns.histplot(genre['energy'],
                            kde=True,
                            )


www_dir = Path(__file__).parent / "www"
app = App(app_ui, server, static_assets=www_dir)
