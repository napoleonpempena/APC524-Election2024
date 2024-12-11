from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
pd.options.mode.chained_assignment = None
import election_dashboard

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('data/president_polls_cleaned.csv')

app = election_dashboard.app_generator(app)
server = app.server

if __name__ == '__main__':
    app.run(debug=True)
