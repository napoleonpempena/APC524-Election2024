import requests
import json
import time
import logging
from requests.exceptions import HTTPError, ConnectionError, Timeout

def fetch_election_data():
    # Replace with the actual API endpoint or data source
    url = "https://api.official-election-site.com/results"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open('election_data.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Election data fetched successfully.")
    except (HTTPError, ConnectionError, Timeout) as e:
        logging.warning(f"Failed to retrieve data due to {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON response: {e}")
        return
    print(f"Error fetching data: {e}")

if __name__ == "__main__":
    while True:
        fetch_election_data()
        time.sleep(300)  # Wait 5 minutes before fetching again

