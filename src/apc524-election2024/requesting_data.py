import pandas as pd

def data_pulling() -> pd.DataFrame:

    ''' Pull data from a remote source. In this case, we use FiveThirtyEight's polling aggregate.

    Returns:
    - df (pd.DataFrame): raw DataFrame with polling data.
    '''

    # Define the URL.
    url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'
    # Pull the data into a Pandas DataFrame.
    df = pd.read_csv(url)

    return df
