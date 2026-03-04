"""
This file runs our full pipeline.

It should run extract -> transform -> load and then return the data for us to look at.
"""

import pandas as pd

from extract import extract
from transform import transform
from load import load
from config import DB_PATH, DEFAULT_BATCH_SIZE





"""
this function is the top-level orchestrator for the full pipeline.

what this function should do:
    call extract() to get raw posts
    call transform() to clean and engineer features
    call load() to write to sqlite
    return the transformed DataFrame
"""

def run_pipeline(subreddit: str = "UCDavis",
                 sort: str = "new",
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 resume: bool = True,
                 db_path: str = DB_PATH) -> pd.DataFrame:
    # TODO: implement
    pass




"""
this function will display a preview of the transformed data.

what this function should do:
    print a summary header
    print a readable preview for each post
    print a footer at the end
    handle empty DataFrames gracefully
"""
def display_output(df: pd.DataFrame) -> None:
    # TODO: implement
    pass




# Scrape 100 new posts, transform them, and load into SQLite
df = run_pipeline(subreddit="UCDavis", sort="new", batch_size=100)

# Start fresh (ignore saved cursor)
df = run_pipeline(subreddit="UCDavis", resume=False)
