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
# import pandas
from datetime import datetime, timezone
import json


# import constants from config.py
from config import (
    HEADERS,
    REDDIT_JSON_URL,
    CURSOR_FILE,
    DEFAULT_BATCH_SIZE,
    MIN_POST_AGE_HOURS,
    PAGE_SLEEP_SECONDS,
)


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
this function will save the cursor point to the disk as a file at the end of every extraction.

this will allow us to resume from the same point, when we run the pipeline again to get another batch of posts

the json responce contains the "after" token. we can access it by response.json()["data"]["after"]

this function should save the "after" token to file CURSOR_FILE (as per config.py)

create a dictionary with the key "after" and the value of the token.
also create a key "last_run", the value should be the current timestamp (when the pipeline ran)
eg. { "after": "t3_1riemy0", "last_run": "2026-03-01T12:00:00" }

write as json to CURSOR_FILE
"""

def save_cursor(after: str | None) -> None:
    cursor_dict = {
        "after": after,
        "last_run": datetime.now().isoformat()
        }
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor_dict, f)
    pass







"""
this function will load and return the cursor from the file CURSOR_FILE

return the value of the "after" key from the json file.

return None if the file is missing or empty.
"""
def load_cursor() -> str | None:
    try:
        with open(CURSOR_FILE, "r") as file:
            data = json.load(file)
            return data.get("after")
    except:
        return None
    pass







"""
this function will check if a post is old enough to be included in our dataset and return a boolean value.
we only want posts that are at least MIN_POST_AGE_HOURS old.

the json responce contains the "created_utc" timestamp as a float (e.g. 1772418658.0)

pandas has a function to convert this to a datetime object: pd.to_datetime(timestamp, unit="s")
"""

def is_old_enough(post_data: dict) -> bool:
    #post_data is a dictionary of post data that we can access from the json responce
    # TODO: implement
    timestamp = post_data.get("created_utc")
    current_time = time.time() #current time in seconds 
    difference = current_time - timestamp # in seconds
    hour = difference/3600 # converting to hours
    if hour >= MIN_POST_AGE_HOURS:
        return True
    return False





"""
this function will fetch the page, and return the raw data and the next cursor.

what this function should do:
    extract the list of children from the json responce. (explore the json file page.json to see how the data is structured)
    each child is a post, and we want to get its data dictioniary.
    get the next cursor
    output a tuple of (list of post dicts, next cursor value)

"""


def fetch_page(subreddit: str = "UCDavis", sort: str = "new",
               limit: int = DEFAULT_BATCH_SIZE,
               after: str | None = None) -> tuple[list[dict], str | None]:

    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    response = requests.get(url, headers = HEADERS, timeout = 10, verify = False).json()
    cursor = response.get('data')['after']
    data = response['data']['children']

    return (data, cursor)

save_cursor(fetch_page()[1])
"""
this function is the orchestrator of the extract stage of our pipeline

it will fetch a batch of posts, filter out posts that are too recent, return a list of posts data dicts and save a new cursor for next time.

print a summary of how many posts were fetched and how many of those passed the age filter
"""

def extract(subreddit: str, sort: str = "new",
            batch_size: int = DEFAULT_BATCH_SIZE,
            resume: bool = True) -> list[dict]:

    cursor = load_cursor()
    fetch = fetch_page(cursor)
    data = fetch[0]
    save_cursor(fetch[1])





    #if resume is True, load the cursor from the file, otherwise set after = None
    #call the other functions here

    #if the fetch function fails, print an error and return an empty list

    # TODO: implement
    pass
