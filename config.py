"""
This file contains the constants for our pipeline

Instead of hard-coding values in other modules, we can import them from here. This makes our code more readable and extensible.

For example, we are using this file to store the regex pattern for the emojis (that we need to remove from titles).

Feel free to add new constants.

"""


import re

# we need a "descriptive user agent" or otherwise reddit will block our request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0)"
}

"""
reddit json api url template
to use: REDDIT_JSON_URL.format(subreddit="UCDavis", sort="new")
"""
REDDIT_JSON_URL = "https://www.reddit.com/r/{subreddit}/{sort}.json"

# DB variables
DB_PATH = "reddit_posts.db"
TABLE_NAME = "posts"


# this is the file that persists after every extract stage, so we can pick up where we left off (not re-scraping the same posts)
CURSOR_FILE = "cursor_state.json"

# emoji regex pattern
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "]+",
    flags=re.UNICODE,
)

# don't change --- reddit's json api caps a single request at 100 posts
DEFAULT_BATCH_SIZE = 100

# min post age
MIN_POST_AGE_HOURS = 12

# sleep time between requests
PAGE_SLEEP_SECONDS = 1
