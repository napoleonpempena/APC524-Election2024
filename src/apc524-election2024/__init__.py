import election_dashboard

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('data/president_polls_cleaned.csv')

app = election_dashboard.data_processor(app, data)
server = app.server

if __name__ == '__main__':
    app.run(debug=True)
