import pytest
import pandas as pd
from src.apc524_election2024.date_data import get_date_data

@pytest.fixture
def sample_data():
    data = {
        "end_date": ["2024-11-04", "2024-11-04", "2024-11-05"],
        "candidate_name": ["Candidate A", "Candidate B", "Candidate A"],
        "votes": [100, 150, 200]
    }
    return pd.DataFrame(data)

def test_get_date_data(sample_data):
    date = "2024-11-04"
    date_df, combined_df = get_date_data(date, sample_data)

    # Test date_df
    assert not date_df.empty
    assert all(date_df["candidate_name"].isin(["Candidate A", "Candidate B"]))
    assert all(date_df["votes"].isin([100, 150]))

    # Test combined_df
    assert not combined_df.empty
    assert list(combined_df["candidate_name"]) == ["Candidate B", "Candidate A"]
    assert list(combined_df["votes"]) == [150, 100]

def test_get_date_data_no_data(sample_data):
    date = "2024-11-06"
    date_df, combined_df = get_date_data(date, sample_data)

    # Test date_df
    assert date_df.empty

    # Test combined_df
    assert combined_df.empty