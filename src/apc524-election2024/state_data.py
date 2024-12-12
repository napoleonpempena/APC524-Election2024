import pandas as pd
from pie_chart import pie_chart
from typing import Tuple


def get_state_data(state: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and processes election data for a specific state from a DataFrame.

    Parameters:
    state (str): The name of the state for which to extract data
    df (pandas.DataFrame): The DataFrame containing election data

    Returns:
    tuple: A tuple containing:
        - state_df (pandas.DataFrame): The DataFrame filtered for the specified state, with the 'state' column dropped and rows with missing values removed.
        - combined_df (pandas.DataFrame): The DataFrame grouped by 'candidate_name', with the sum of votes, sorted by votes in descending order, and containing only 'candidate_name' and 'votes' columns.
    """
    state_df = df[df["state"] == state]
    state_df = state_df.drop(columns=["state"])
    state_df = state_df.dropna()

    combined_df = state_df.groupby("candidate_name").sum().reset_index()
    combined_df = combined_df.sort_values(by="votes", ascending=False)
    combined_df = combined_df[["candidate_name", "votes"]]

    return state_df, combined_df


if __name__ == "__main__":
    df = pd.read_csv("data/president_polls_cleaned.csv")
    state = "Texas"

    state_df, combined_df = get_state_data(state, df)

    pie_chart(combined_df, state=state)
