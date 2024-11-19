import pandas as pd
import matplotlib.pyplot as plt


def get_state_data(state, df):
    """
    Extracts and processes election data for a specific state from a DataFrame.

    Parameters:
    state (str): The name of the state for which to extract data.
    df (pandas.DataFrame): The DataFrame containing election data with columns including 'state', 'candidate_name', and 'votes'.

    Returns:
    tuple: A tuple containing:
        - state_df (pandas.DataFrame): The DataFrame filtered for the specified state, with the 'state' column dropped and rows with missing values removed.
        - combined_data (pandas.DataFrame): The DataFrame grouped by 'candidate_name', with the sum of votes, sorted by votes in descending order, and containing only 'candidate_name' and 'votes' columns.
    """
    state_df = df[df["state"] == state]
    state_df = state_df.drop(columns=["state"])
    state_df = state_df.dropna()

    combined_data = state_df.groupby("candidate_name").sum().reset_index()
    combined_data = combined_data.sort_values(by="votes", ascending=False)
    combined_data = combined_data[["candidate_name", "votes"]]

    return state_df, combined_data


if __name__ == "__main__":
    df = pd.read_csv("data/president_polls_cleaned.csv")

    state_df, combined_data = get_state_data("New Jersey", df)

    # Make pie chart of combined data
    plt.figure(figsize=(10, 10))
    plt.pie(
        combined_data["votes"],
        labels=combined_data["candidate_name"],
        autopct="%1.1f%%",
    )
    plt.title("New Jersey Poll Results")
    plt.show()
