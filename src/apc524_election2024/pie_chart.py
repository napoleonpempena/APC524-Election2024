import pandas as pd
import matplotlib.pyplot as plt


def pie_chart(combined_df=None, state=None, date=None) -> None:
    """
    Generates a pie chart for poll results.

    This function takes a DataFrame containing candidate names and their respective votes,
    combines candidates with less than 2% of the total votes into an 'Other' category,
    and then generates a pie chart displaying the vote distribution.

    Parameters:
    combined_data (pd.DataFrame): A DataFrame with columns 'candidate_name' and 'votes'
                                  representing the candidates and their respective vote counts.

    Returns:
    None: The function displays a pie chart and does not return any value.
    """

    if state is None:
        loc_str = "the Entire Country"
    else:
        loc_str = state

    if date is None:
        date_str = "Across All Dates"
    else:
        date_str = f"on {date}"

    threshold: float = 0.02  # 2% threshold
    total_votes: int = combined_df["votes"].sum()
    combined_df["percentage"]: pd.Series = combined_df["votes"] / total_votes

    major_candidates: pd.DataFrame = combined_df[combined_df["percentage"] >= threshold]
    other_candidates: pd.DataFrame = combined_df[combined_df["percentage"] < threshold]

    other_votes: int = other_candidates["votes"].sum()
    other_row: pd.DataFrame = pd.DataFrame(
        [["Other", other_votes]], columns=["candidate_name", "votes"]
    )

    final_data: pd.DataFrame = pd.concat(
        [major_candidates, other_row], ignore_index=True
    )

    plt.figure(figsize=(10, 10))
    plt.pie(
        final_data["votes"],
        labels=final_data["candidate_name"],
        autopct="%1.1f%%",
    )
    plt.title(f"Poll Results for {loc_str} {date_str}")
    plt.show()
    return
