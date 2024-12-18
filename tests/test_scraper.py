import os
from unittest.mock import patch, Mock
from apc524_election2024.scraper import build_path, pour_soup, main


def test_build_path_national():
    site = "http://example.com"
    poll_type = "presidential"
    jurisdiction = "national"
    expected_path = os.path.join(site, "president/general/2024/trump-vs-harris")
    assert build_path(site, poll_type, jurisdiction) == expected_path


def test_build_path_state():
    site = "http://example.com"
    poll_type = "presidential"
    jurisdiction = "california"
    expected_path = os.path.join(
        site, "president/general/2024/california/trump-vs-harris"
    )
    assert build_path(site, poll_type, jurisdiction) == expected_path


@patch("src.apc524_election2024.scraper.requests.get")
def test_pour_soup(mock_get):
    mock_response = Mock()
    mock_response.content = "<html><head><title>Test</title></head><body></body></html>"
    mock_get.return_value = mock_response

    path = "http://example.com/president/general/2024/trump-vs-harris"
    soup = pour_soup(path)
    assert soup.title.string == "Test"


@patch("src.apc524_election2024.scraper.pour_soup")
def test_main(mock_pour_soup):
    mock_soup = Mock()
    mock_pour_soup.return_value = mock_soup

    sitename = "http://example.com"
    poll_type = "presidential"
    jurisdiction = "national"
    result = main(sitename, poll_type, jurisdiction)
    assert result == mock_soup
    mock_pour_soup.assert_called_once_with(
        os.path.join(sitename, "president/general/2024/trump-vs-harris")
    )
