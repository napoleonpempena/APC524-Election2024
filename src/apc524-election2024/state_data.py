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
    state = "Texas"

    state_df, combined_data = get_state_data(state, df)

    # Combine small percentage candidates into 'Other' category
    threshold = 0.02  # 5% threshold
    total_votes = combined_data["votes"].sum()
    combined_data["percentage"] = combined_data["votes"] / total_votes

    # Separate major candidates and others
    major_candidates = combined_data[combined_data["percentage"] >= threshold]
    other_candidates = combined_data[combined_data["percentage"] < threshold]

    # Sum votes for 'Other' category
    other_votes = other_candidates["votes"].sum()
    other_row = pd.DataFrame(
        [["Other", other_votes]], columns=["candidate_name", "votes"]
    )

    # Combine major candidates with 'Other' category
    final_data = pd.concat([major_candidates, other_row], ignore_index=True)

    # Make pie chart of final data
    plt.figure(figsize=(10, 10))
    plt.pie(
        final_data["votes"],
        labels=final_data["candidate_name"],
        autopct="%1.1f%%",
    )
    plt.title(f"{state} Poll Results")
    plt.show()
