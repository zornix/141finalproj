"""
This file contains the extract stage of our pipeline

We will pull raw post data using Reddit's public JSON API.

The core functionality is:
    - fetch posts in batches of up to 100 (as per reddit's rate limits)
    - filter out posts younger than MIN_POST_AGE_HOURS (defined in config.py)
    - track where we left off in the scraping process by saving the cursor pointer to a file at the end of every extraction
        (CURSOR_FILE, defined in config.py)
"""


from wsgiref import headers

import requests
import urllib3
import time
import json
import os
import pandas as pd
from datetime import datetime, timezone
import json


# import constants from config.py
from config import (
    HEADERS,
    REDDIT_JSON_URL,
    CURSOR_FILE,
    MIN_POST_AGE_HOURS,
)


#
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# This function saves the last cursor and the current time to a local file.

def save_cursor(after: str | None) -> None:
    cursor_dict = {
        "after": after,
        "last_run": datetime.now().isoformat()
        }
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor_dict, f)
    pass




# This function loads and returns the value of the cursor token from the file. Returns None if the file is empty or missing.

def load_cursor() -> str | None:
    try:
        with open(CURSOR_FILE, "r") as file:
            data = json.load(file)
            return data.get("after")
    except:
        return None
    pass



# This function checks if a post is old enough to be included in our dataset.

def is_old_enough(post_data: dict) -> bool:
    #post_data is a dictionary of post data that we can access from the json responce
    timestamp = post_data.get("created_utc")
    current_time = time.time() #current time in seconds
    difference = current_time - timestamp # in seconds
    hour = difference/3600 # converting to hours
    if hour >= MIN_POST_AGE_HOURS:
        return True # we only want posts that are at least MIN_POST_AGE_HOURS old.
    return False




# This function fetches 100 posts from the UCDavis subreddit, returns a tuple of a list of post data dictionaries and the cursor token.

def fetch_page(after: str | None = None) -> tuple[list[dict], str | None]:
    if after is not None:
      url = REDDIT_JSON_URL + after
    else:
      url = REDDIT_JSON_URL
    response = requests.get(url, headers = HEADERS, timeout = 10, verify = False).json()
    cursor = response.get('data')['after']
    data = response['data']['children']

    return (data, cursor)




# This function orchestrates the whole extract stage. It fetches a batch of posts, filters out those that are too recent and return a list of valid post data distionaries.

def extract(resume: bool = True) -> list[dict]:
    try:
        if resume:
            cursor = load_cursor()
            fetch = fetch_page(cursor)
            listdata = fetch[0]
            save_cursor(fetch[1])
        else:
            fetch = fetch_page()
            listdata = fetch[0]
            save_cursor(fetch[1])
    except:
        print("Error")
        return []


    data = [post for post in listdata if is_old_enough(post['data'])]


    return data
