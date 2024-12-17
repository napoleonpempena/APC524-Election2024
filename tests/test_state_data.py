import pandas as pd
from src.apc524_election2024.state_data import get_state_data


def test_get_state_data():
    data = {
        "state": ["Texas", "Texas", "California", "Texas", "California"],
        "candidate_name": ["A", "B", "A", "C", "B"],
        "votes": [100, 150, 200, 250, 300],
    }
    df = pd.DataFrame(data)

    state = "Texas"
    state_df, combined_df = get_state_data(state, df)

    expected_state_df = pd.DataFrame(
        {"candidate_name": ["A", "B", "C"], "votes": [100, 150, 250]}
    )

    expected_combined_df = pd.DataFrame(
        {"candidate_name": ["C", "B", "A"], "votes": [250, 150, 100]}
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(state_df.reset_index(drop=True), expected_state_df)
    pd.testing.assert_frame_equal(
        combined_df.reset_index(drop=True), expected_combined_df
    )


def test_get_state_data_no_data():
    data = {
        "state": ["California", "California"],
        "candidate_name": ["A", "B"],
        "votes": [200, 300],
    }
    df = pd.DataFrame(data)

    state = "Texas"
    state_df, combined_df = get_state_data(state, df)

    expected_state_df = pd.DataFrame(columns=["candidate_name", "votes"])
    expected_combined_df = pd.DataFrame(columns=["candidate_name", "votes"])

    pd.testing.assert_frame_equal(state_df, expected_state_df)
    pd.testing.assert_frame_equal(combined_df, expected_combined_df)
