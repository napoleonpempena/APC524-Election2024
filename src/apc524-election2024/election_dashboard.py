from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
pd.options.mode.chained_assignment = None
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

def datetime_assignment(data):
    ''' Change columns with datetimes to datetime data types. '''
    date_columns = ['start_date', 'end_date']
    data[date_columns] = data[date_columns].apply(pd.to_datetime)
    return data

def state_nan_cleaner(data):
    data['state'] = data['state'].fillna("National")
    return data

def likely_candidates(data,
                       N: int=5):

    data['pct'] = data['pct'].astype(float)
    candidates = data[['candidate_name', 'pct']].groupby('candidate_name').sum().sort_values(by    ='pct', ascending=False).head(N).index.values
    data = data.loc[data['candidate_name'].isin(candidates)]

    return data

def data_processor(app, data):

    data = pd.read_csv('data/president_polls_cleaned.csv')
    # Convert data columns with dates to datetime objects
    df = datetime_assignment(data)
    # Filter out unlikely candidates
    df = likely_candidates(df)
    # Convert 'NaN' values to 'Other'
    df = state_nan_cleaner(df)
    # Turn date range to Unix second convention
    start_date_seconds, end_date_seconds = [pd.Timestamp(df['start_date'].min()).timestamp(),
                                            pd.Timestamp(df['end_date'].max()).timestamp()]
    # Define intervals for slider mark ticks
    step_interval = 86400 * 30 # approximately 1 month
    # Define intervals for slider mark tick labels
    mark_interval = df[['start_date', 'end_date']].resample('6M', on='start_date').min().index

    app.layout = html.Div([
        dcc.Store(id='memory-output'),
        html.Div([
            dcc.Checklist(df['candidate_name'].sort_values().unique(),
                        ['Kamala Harris', 'Donald Trump'],
                        id='candidate_name-checkbox'),
        ], style={'float': 'left', 'width': '33%', 'display': 'inline-block'}),

        html.Div([
        dcc.Dropdown(
            df['state'].sort_values().unique(),
            'National',
            id='state_name-dropdown')
        ], style={'float': 'right', 'width': '66%', 'display': 'inline-block'}),

        html.Div([
            dcc.Checklist(
                ['All polls', 'Composite polling average'],
                ['Composite polling average'],
                inline=True,
                id='data_display-checkbox')
        ], style={'float': 'right', 'width': '66%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='graph-with-slider'),
            dcc.RangeSlider(min=start_date_seconds,
                            max=end_date_seconds,
                            step=step_interval,
                            value=[start_date_seconds, end_date_seconds],
                            marks={int(d): pd.to_datetime(d, unit='s').strftime('%Y-%m')
                                for d in mark_interval.map(pd.Timestamp.timestamp)},
                            id='date_range-slider'),
        ], style={'width': '100%', 'display': 'inline-block'})
    ])
    return app, df

@callback(
    Output('graph-with-slider', 'figure'),
    Input('memory-output', 'data'),
    Input('candidate_name-checkbox', 'value'),
    Input('state_name-dropdown', 'value'),
    Input('data_display-checkbox', 'value'),
    Input('date_range-slider', 'value'))
def update_figure(data, candidate_names, state_name, data_display, date_values):

    print(data)

    df = pd.DataFrame(data)

    print(df)

    ''' Date filtering. '''
    start, end = date_values
    df['start_date-unix'] = df['start_date'].apply(pd.Timestamp.timestamp)
    # Filter data by date values
    filtered_df = df[(df['start_date-unix'] >= start) & (df['start_date-unix'] <= end)]

    ''' Candidate filtering. '''
    filtered_df = filtered_df[filtered_df['candidate_name'].isin(candidate_names)]
    ''' State filtering. '''
    filtered_df = filtered_df[filtered_df['state'] == state_name]

    ''' Calculate composite monthly averages. '''
    composite_monthly_average = filtered_df.groupby('candidate_name').resample('W', on='start_date')[['votes', 'pct']].mean()
    months = pd.date_range(start=filtered_df['start_date'].min(), end=filtered_df['start_date'].max(), freq='W')
    composite_monthly_average_reindexed = composite_monthly_average.reindex(months, level=1).ffill()
    composite_monthly_average_reindexed = composite_monthly_average_reindexed.reset_index(level=0).reset_index()

    if data_display == ['All polls']:
        fig = px.scatter(filtered_df, x='start_date', y='pct', color='candidate_name')
    elif data_display == ['Composite polling average']:
        fig = px.line(composite_monthly_average_reindexed, x='start_date', y='pct', color='candidate_name', markers=True)
    elif sorted(data_display) == ['All polls', 'Composite polling average']:
        fig = px.scatter(filtered_df, x='start_date', y='pct', color='candidate_name')
        for entry_index, entry in enumerate(filtered_df['candidate_name'].unique()):
            candidate_entry = composite_monthly_average_reindexed.loc[composite_monthly_average_reindexed['candidate_name'] == entry]
            fig.add_scatter(x=candidate_entry['start_date'],
                            y=candidate_entry['pct'],
                            mode='lines',
                            name=entry,
                            line=go.scatter.Line(color=PLOTLY_COLORS[entry_index]),)
    else:
        fig = px.scatter()


    fig.update_layout()

    return fig
