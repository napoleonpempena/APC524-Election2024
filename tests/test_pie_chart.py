import pandas as pd
import pytest
from apc524_election2024.pie_chart import pie_chart


def test_pie_chart_no_state_no_date(mocker):
    data = {
        "candidate_name": ["Candidate A", "Candidate B", "Candidate C", "Candidate D"],
        "votes": [5000, 3000, 1000, 500],
    }
    df = pd.DataFrame(data)

    mocker.patch("matplotlib.pyplot.show")

    pie_chart(df)

    assert True  # If no exception is raised, the test passes


def test_pie_chart_with_state_and_date(mocker):
    data = {
        "candidate_name": ["Candidate A", "Candidate B", "Candidate C", "Candidate D"],
        "votes": [5000, 3000, 1000, 500],
    }
    df = pd.DataFrame(data)

    mocker.patch("matplotlib.pyplot.show")

    pie_chart(df, state="California", date="2024-11-05")

    assert True  # If no exception is raised, the test passes


def test_pie_chart_with_other_category(mocker):
    data = {
        "candidate_name": [
            "Candidate A",
            "Candidate B",
            "Candidate C",
            "Candidate D",
            "Candidate E",
        ],
        "votes": [5000, 3000, 1000, 500, 100],
    }
    df = pd.DataFrame(data)

    mocker.patch("matplotlib.pyplot.show")

    pie_chart(df)

    assert True  # If no exception is raised, the test passes

@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_pie_chart_empty_dataframe(mocker):
    df = pd.DataFrame(columns=["candidate_name", "votes"])

    mocker.patch("matplotlib.pyplot.show")

    pie_chart(df)

    assert True  # If no exception is raised, the test passes
