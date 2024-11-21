import pandas as pd


def test_data_cleaning():
    # Assuming the data_cleaning.py script has been executed and the cleaned file is saved
    cleaned_df = pd.read_csv("data/president_polls_cleaned.csv")
    # Check if the required columns are present
    required_columns = ["sample_size", "pct", "votes", "start_date", "end_date"]
    for column in required_columns:
        assert column in cleaned_df.columns
    # Check if unnecessary columns are removed
    unnecessary_columns = [
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
    for column in unnecessary_columns:
        assert column not in cleaned_df.columns
