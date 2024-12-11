import os
import pandas
from dash import Dash
import election_dashboard

# Application starter
def initialize_application():
    # Import stylesheet
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    # Initialize Dash application
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    return app

def main(app):
    # Import data
    data = pd.read_csv('president_polls_cleaned.csv')
    # Process the data
    app = election_dashboard.data_processor(app, data)

    return app

if __name__ == "__main__":
    app = initialize_application()
    server = app.server

    app = main(app)
    app.run(debug=True)
