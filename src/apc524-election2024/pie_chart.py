import pandas as pd
# import matplotlib.pyplot as plt
import plotly.express as px

# Global constants for plotting
FIG_WIDTH = 600 # units of pixels
FIG_HEIGHT = FIG_WIDTH

def pie_chart(combined_df: pd.DataFrame, 
              state: str | None=None, 
              date=None, 
              fig_backend: str='plotly'):
    """
    Generates a pie chart for poll results.

    This function takes a DataFrame containing candidate names and their respective votes,
    combines candidates with less than 2% of the total votes into an 'Other' category,
    and then generates a pie chart displaying the vote distribution.

    Parameters:
    combined_data (pd.DataFrame): A DataFrame with columns 'candidate_name' and 'votes'
                                  representing the candidates and their respective vote counts.
    fig_backend (str):            String denoting whether visualization backend is `matplotlib` or `plotly`.
    Returns:
    None: The function displays a pie chart and does not return any value.
    """

    # Ensure that a valid backend data type is provided
    assert isinstance(fig_backend, str)

    if state is None:
        loc_str = "the Entire Country"
    else:
        loc_str = state

    if date is None:
        date_str = "Across All Dates"
    else:
        date_str = f"on {date}"

    threshold = 0.02  # 2% threshold
    total_votes = combined_df["votes"].sum()
    combined_df["percentage"] = combined_df["votes"] / total_votes

    major_candidates = combined_df[combined_df["percentage"] >= threshold]
    other_candidates = combined_df[combined_df["percentage"] < threshold]

    other_votes = other_candidates["votes"].sum()
    other_row = pd.DataFrame(
        [["Other", other_votes]], columns=["candidate_name", "votes"]
    )

    final_data = pd.concat([major_candidates, other_row], ignore_index=True)

    # Define the plot title
    plot_title = f"Poll Results for {loc_str} {date_str}"
    # Plot the data using the selected backend
    if fig_backend == 'matplotlib':
        # GAR: commenting out matplotlib code on server due to tight data constraints
        # plt.figure(figsize=(10, 10))
        # plt.pie(
        #     final_data["votes"],
        #     labels=final_data["candidate_name"],
        #     autopct="%1.1f%%",
        # )
        # plt.title(plot_title)
        # plt.show()

        return
    else:
        fig = px.pie(final_data, 
                     values="votes", 
                     names="candidate_name",
                     title=plot_title,
                     width=FIG_WIDTH,
                     height=FIG_HEIGHT)
        return fig
