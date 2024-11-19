import pandas as pd
from pie_chart import pie_chart


def get_date_data(date, df):
    """
    Extracts and processes data for a specific date from election DataFrame.

    Parameters:
    date (str): The date for which data is to be extracted.
    df (pandas.DataFrame): The DataFrame containing the election data.

    Returns:
    tuple: A tuple containing:
        - date_df (pandas.DataFrame): The DataFrame filtered by the specified date
        - combined_df (pandas.DataFrame): The DataFrame grouped by 'candidate_name' with relevant voting data
    """
    date_df = df[df["end_date"] == date]
    date_df = date_df.drop(columns=["end_date"])
    date_df = date_df.dropna()

    combined_df = date_df.groupby("candidate_name").sum().reset_index()
    combined_df = combined_df.sort_values(by="votes", ascending=False)
    combined_df = combined_df[["candidate_name", "votes"]]

    return date_df, combined_df


if __name__ == "__main__":
    df = pd.read_csv("data/president_polls_cleaned.csv")
    date = "2024-11-04"

    date_df, combined_df = get_date_data(date, df)

    print(combined_df)
    pie_chart(combined_df, date=date)
