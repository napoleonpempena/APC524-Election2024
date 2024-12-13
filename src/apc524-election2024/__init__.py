import dash
from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
pd.options.mode.chained_assignment = None

# Local imports
import requesting_data
import data_cleaning
import election_dashboard

# Pull data from remote location
data = requesting_data.data_pulling()
# Clean the data and save it locally
df = data_cleaning.data_pruning(data, save_csv=True)
# Link to external Bootstrap stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Initialize Dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Fill Dash application with HTML layout and dynamic callbacks
app = election_dashboard.generate_app(app)
# Initialize application server for Heroku integration
server = app.server

# Run the application platform
if __name__ == '__main__':
    app.run(debug=True)
