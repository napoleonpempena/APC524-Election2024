import os
import requests
from bs4 import BeautifulSoup

def build_path(site, poll_type, jurisdiction):
    ''' Builds a path given a polling website, a poll type, and a poll jurisdiction. '''    
    jurisdiction = jurisdiction.lower()
    if jurisdiction.lower() == "national":
        specifier = "president/general/2024/trump-vs-harris"
    else:
        specifier = "president/general/2024/{0}/trump-vs-harris".format(jurisdiction)

    path = os.path.join(site, specifier)

    return path

def pour_soup(path):
    ''' Creates URL request and scrapes page for a given path. '''
    page = requests.get(path)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup

def main(sitename, poll_type="presidential", jurisdiction="national"):
    path = build_path(sitename, poll_type, jurisdiction)
    soup = pour_soup(path)
    return soup
