import pandas as pd

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
