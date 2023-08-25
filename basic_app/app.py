from shiny import App, render, ui

app_ui = ui.page_fluid(
    ui.h2("Hello world of Recommender Systems!"),
    ui.input_slider("n", "N", 0, 140, 20),
    ui.output_text("txt"),
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n*2 is {input.n() * 2}"


app = App(app_ui, server)
