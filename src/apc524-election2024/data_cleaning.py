import pandas as pd
import numpy as np

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

def data_pruning(df: pd.DataFrame,
                 save_csv: bool=True) -> pd.DataFrame:

    ''' Method to simplify DataFrame to minimize computational expense and remote memory loading.
    Allows saving if boolean enabled for it.

    Args:
    - df (pd.DataFrame): raw Pandas DataFrame
    Returns:
    - df (pd.DataFrame): filtered and simplified Pandas DataFrame
    '''

    # Drop columns irrelevant to the analysis
    df = df.drop(
        columns=[
            "question_id",
            "poll_id",
            "cycle",
            "pollster_id",
            "sponsors",
            "sponsor_ids",
            "display_name",
            "pollster_rating_id",
            "pollster_rating_name",
            "population",
            "population_full",
            "methodology",
            "office_type",
            "seat_number",
            "seat_name",
            "sponsor_candidate_id",
            "sponsor_candidate",
            "internal",
            "partisan",
            "tracking",
            "nationwide_batch",
            "created_at",
            "notes",
            "url",
            "url_article",
            "url_topline",
            "url_crosstab",
            "subpopulation",
            "numeric_grade",
            "pollscore",
            "transparency_score",
            "sponsor_candidate_party",
            "endorsed_candidate_id",
            "endorsed_candidate_name",
            "endorsed_candidate_party",
            "source",
            "race_id",
            "election_date",
            "stage",
            "ranked_choice_reallocated",
            "ranked_choice_round",
            "hypothetical",
            "candidate_id",
        ]
    )

    # Add column for total voters (pct/100 * sample size)
    df["votes"] = np.round(df["sample_size"] * df["pct"] / 100)

    # Save data, if boolean enabled
    if save_csv:
        df.to_csv("data/president_polls_cleaned.csv", index=False)

    return df

