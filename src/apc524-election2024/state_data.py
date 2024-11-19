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


def state_pie_chart(combined_data):
    """
    Generates a pie chart for state poll results.

    This function takes a DataFrame containing candidate names and their respective votes,
    combines candidates with less than 2% of the total votes into an 'Other' category,
    and then generates a pie chart displaying the vote distribution.

    Parameters:
    combined_data (pd.DataFrame): A DataFrame with columns 'candidate_name' and 'votes'
                                  representing the candidates and their respective vote counts.

    Returns:
    None: The function displays a pie chart and does not return any value.
    """

    threshold = 0.02  # 5% threshold
    total_votes = combined_data["votes"].sum()
    combined_data["percentage"] = combined_data["votes"] / total_votes

    major_candidates = combined_data[combined_data["percentage"] >= threshold]
    other_candidates = combined_data[combined_data["percentage"] < threshold]

    other_votes = other_candidates["votes"].sum()
    other_row = pd.DataFrame(
        [["Other", other_votes]], columns=["candidate_name", "votes"]
    )

    final_data = pd.concat([major_candidates, other_row], ignore_index=True)

    plt.figure(figsize=(10, 10))
    plt.pie(
        final_data["votes"],
        labels=final_data["candidate_name"],
        autopct="%1.1f%%",
    )
    plt.title(f"{state} Poll Results")
    plt.show()
    return


if __name__ == "__main__":
    df = pd.read_csv("data/president_polls_cleaned.csv")
    state = "Texas"

    state_df, combined_data = get_state_data(state, df)
    state_pie_chart(combined_data)
