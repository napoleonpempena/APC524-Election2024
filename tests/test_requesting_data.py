import pytest
import requests
import json
import logging
from unittest.mock import patch, mock_open
from requests.exceptions import HTTPError, ConnectionError, Timeout
from src.apc524_election2024.requesting_data import fetch_election_data

@patch('requests.get')
@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_fetch_election_data_success(mock_json_dump, mock_open, mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"result": "success"}

    fetch_election_data()

    mock_get.assert_called_once_with("https://api.official-election-site.com/results")
    mock_response.raise_for_status.assert_called_once()
    mock_response.json.assert_called_once()
    mock_open.assert_called_once_with('election_data.json', 'w')
    mock_json_dump.assert_called_once_with({"result": "success"}, mock_open(), indent=4)

@patch('requests.get', side_effect=HTTPError("HTTP Error"))
def test_fetch_election_data_http_error(mock_get):
    with patch('logging.warning') as mock_logging_warning:
        fetch_election_data()
        mock_logging_warning.assert_called_once_with("Failed to retrieve data due to HTTP Error")

@patch('requests.get', side_effect=ConnectionError("Connection Error"))
def test_fetch_election_data_connection_error(mock_get):
    with patch('logging.warning') as mock_logging_warning:
        fetch_election_data()
        mock_logging_warning.assert_called_once_with("Failed to retrieve data due to Connection Error")

@patch('requests.get', side_effect=Timeout("Timeout Error"))
def test_fetch_election_data_timeout_error(mock_get):
    with patch('logging.warning') as mock_logging_warning:
        fetch_election_data()
        mock_logging_warning.assert_called_once_with("Failed to retrieve data due to Timeout Error")

@patch('requests.get')
def test_fetch_election_data_json_decode_error(mock_get):
    mock_response = mock_get.return_value
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)

    with patch('logging.error') as mock_logging_error:
        fetch_election_data()
        mock_logging_error.assert_called_once_with("Failed to decode JSON response: Expecting value: line 1 column 1 (char 0)")