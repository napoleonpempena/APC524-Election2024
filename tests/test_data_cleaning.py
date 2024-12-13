from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

@pytest.fixture
def sample_data():
    data = {
        "sample_size": [1000, 1500, 2000],
        "pct": [50, 60, 70],
        "start_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "end_date": ["2023-01-10", "2023-02-10", "2023-03-10"],
        "question_id": [1, 2, 3],
        "poll_id": [1, 2, 3],
        "cycle": [2024, 2024, 2024],
        "pollster_id": [1, 2, 3],
        "sponsors": ["A", "B", "C"],
        "sponsor_ids": [1, 2, 3],
        "display_name": ["Poll A", "Poll B", "Poll C"],
        "pollster_rating_id": [1, 2, 3],
        "pollster_rating_name": ["A", "B", "C"],
        "population": ["lv", "rv", "a"],
        "population_full": ["Likely Voters", "Registered Voters", "Adults"],
        "methodology": ["phone", "online", "mixed"],
        "office_type": ["president", "president", "president"],
        "seat_number": [1, 1, 1],
        "seat_name": ["President", "President", "President"],
        "sponsor_candidate_id": [1, 2, 3],
        "sponsor_candidate": ["A", "B", "C"],
        "internal": [False, False, False],
        "partisan": [False, False, False],
        "tracking": [False, False, False],
        "nationwide_batch": [False, False, False],
        "created_at": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "notes": ["", "", ""],
        "url": ["", "", ""],
        "url_article": ["", "", ""],
        "url_topline": ["", "", ""],
        "url_crosstab": ["", "", ""],
        "subpopulation": ["", "", ""],
        "numeric_grade": ["", "", ""],
        "pollscore": ["", "", ""],
        "transparency_score": ["", "", ""],
        "sponsor_candidate_party": ["", "", ""],
        "endorsed_candidate_id": [1, 2, 3],
        "endorsed_candidate_name": ["A", "B", "C"],
        "endorsed_candidate_party": ["", "", ""],
        "source": ["", "", ""],
        "race_id": [1, 2, 3],
        "election_date": ["2024-11-05", "2024-11-05", "2024-11-05"],
        "stage": ["", "", ""],
        "ranked_choice_reallocated": [False, False, False],
        "ranked_choice_round": [1, 1, 1],
        "hypothetical": [False, False, False],
        "candidate_id": [1, 2, 3],
    }
    return pd.DataFrame(data)

def test_data_cleaning(sample_data: pd.DataFrame, tmp_path: Path):
    # Save sample data to a temporary CSV file
    sample_data_path = tmp_path / "president_polls.csv"
    sample_data.to_csv(sample_data_path, index=False)

    # Run the data cleaning script
    df = pd.read_csv(sample_data_path)

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

    df["votes"] = np.round(df["sample_size"] * df["pct"] / 100).astype(int)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])

    # Expected DataFrame after cleaning
    expected_data = {
        "sample_size": [1000, 1500, 2000],
        "pct": [50, 60, 70],
        "start_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        "end_date": pd.to_datetime(["2023-01-10", "2023-02-10", "2023-03-10"]),
        "votes": [500, 900, 1400],
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert the cleaned DataFrame matches the expected DataFrame
    assert_frame_equal(df, expected_df)