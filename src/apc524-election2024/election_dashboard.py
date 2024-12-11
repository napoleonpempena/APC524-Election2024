from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd
pd.options.mode.chained_assignment = None
import plotly.express as px
import plotly.graph_objects as go

# Set global color scheme for consistent colormap
PLOTLY_COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

def datetime_assignment(data: pd.DataFrame) -> pd.DataFrame:

    ''' Change columns with datetimes to datetime data types.

    Args:
    - data (pandas DataFrame): DataFrame with datetime values
    Returns:
    - data (pandas DataFrame): DataFrame with re-typed datetime values

    '''

    # Columns over which to look for date data
    date_columns = ['start_date', 'end_date']
    # Change data types
    data[date_columns] = data[date_columns].apply(pd.to_datetime)

    return data

def state_nan_cleaner(data: pd.DataFrame) -> pd.DataFrame:

    ''' Function to filter out nan values and replace them with representative "National" label.

    Args:
    - data (pandas DataFrame): DataFrame with state names and nan values
    Returns:
    - data (pandas DataFrame): DataFrame with state names and filtered values
    '''

    data['state'] = data['state'].fillna("National")
    return data

def likely_candidates(data: pd.DataFrame, N: int=5) -> pd.DataFrame:

    ''' Function to pare down number of candidates based on polling composites.

    Args:
    - data (pandas DataFrame): DataFrame with polling percentage data and candidate names.
    - N (int): number of candidates to pare down to - the N candidates with highest polling composites will remain.
    Returns:
    - data (pandas DataFrame): DataFrame with polling data for N candidates with highest polling composites.

    '''

    # Ensure polling percentages are numerics
    data['pct'] = data['pct'].astype(float)
    # Group data by candidate names and get poll data sums as crude metric for presidential viability. Only grab the top "N".
    candidates = data[['candidate_name', 'pct']].groupby('candidate_name').sum().sort_values(by    ='pct', ascending=False).head(N).index.values
    # Get candidate names
    data = data.loc[data['candidate_name'].isin(candidates)]

    return data

#####################################################################################
# Begin data processing.
#  Note that Dash and Plotly function seamlessly with global variables in this script. 
#####################################################################################

# Load data 
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

''' End data processing. '''

def generate_app(app):

    ''' Method to generate Dash app HTML layout and callback functions.

    Args:
    - app (Dash application): Dash application object
    Returns:
    - app (Dash application): Dash application object with set HTML elements and callbacks
    '''

    # Set HTML wrapper
    app.layout = html.Div([
        # Create checklist with top N candidates, as filtered by likely_candidates()
        # Preset Kamala Harris and Donald Trump
        html.Div([
            dcc.Checklist(df['candidate_name'].sort_values().unique(),
                        ['Kamala Harris', 'Donald Trump'],
                        id='candidate_name-checkbox'),
        ], style={'float': 'left', 'width': '33%', 'display': 'inline-block'}),
        # Create dropdown list with all states to allow for statewise filtering
        html.Div([
        dcc.Dropdown(
            df['state'].sort_values().unique(),
            'National',
            id='state_name-dropdown')
        ], style={'float': 'right', 'width': '66%', 'display': 'inline-block'}),
        # Create checklist to let user decide data display method
        html.Div([
            dcc.Checklist(
                ['All polls', 'Composite polling average'],
                ['Composite polling average'],
                inline=True,
                id='data_display-checkbox')
        ], style={'float': 'right', 'width': '66%', 'display': 'inline-block'}),
        # Create date slider to allow user to filter by polling date
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

    return app

@callback(
    Output('graph-with-slider', 'figure'),
    Input('candidate_name-checkbox', 'value'),
    Input('state_name-dropdown', 'value'),
    Input('data_display-checkbox', 'value'),
    Input('date_range-slider', 'value'))
def update_figure(candidate_names, state_name, data_display, date_values):

    ''' Method to allow dynamic updates to the output figure based on user inputs.

    Args: Dash inputs from interactive elements defined in the Dash application layout.
    '''

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

    # Conditional filtering based on user inputs - poll scatter and trendline formatting features
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
