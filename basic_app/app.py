from shiny import App, render, ui
from matplotlib import pyplot as plt

app_ui = ui.page_fluid(
    ui.h2("Hello world of Rec Systems"),
    ui.input_slider("n", "N", 0, 140, 20),
    ui.output_text("txt"),
    ui.output_plot("test_plot"),
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"
    @output
    @render.plot
    def test_plot():
        return plt.scatter([1,2,3],[4,4,7])

app = App(app_ui, server)
