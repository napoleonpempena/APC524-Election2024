import os
import requests
from bs4 import BeautifulSoup


def build_path(site: str, poll_type: str, jurisdiction: str) -> str:
    """Builds a path given a polling website, a poll type, and a poll jurisdiction."""
    jurisdiction = jurisdiction.lower()
    if jurisdiction.lower() == "national":
        specifier = "president/general/2024/trump-vs-harris"
    else:
        specifier = f"president/general/2024/{jurisdiction}/trump-vs-harris"

    path = os.path.join(site, specifier)

    return path


def pour_soup(path: str) -> BeautifulSoup:
    """Creates URL request and scrapes page for a given path."""
    page = requests.get(path)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup


def main(
    sitename: str, poll_type: str = "presidential", jurisdiction: str = "national"
) -> BeautifulSoup:
    path = build_path(sitename, poll_type, jurisdiction)
    soup = pour_soup(path)
    return soup
