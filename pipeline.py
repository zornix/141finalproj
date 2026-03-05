"""
This file runs our full pipeline.

It should run extract -> transform -> load and then return the data for us to look at.
"""

import pandas as pd

from extract import extract
from transform import transform
from load import load
from config import DB_PATH, DEFAULT_BATCH_SIZE


# This function is the top-level orchestrator for the full pipeline.
def run_pipeline(subreddit: str = "UCDavis",
                 sort: str = "new",
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 resume: bool = True,
                 db_path: str = DB_PATH) -> pd.DataFrame:
    raw_posts = extract() # Gets the raw posts
    transformed_df = transform(raw_posts) # Clean and engineer features
    load(transformed_df, DB_PATH) # Write sqlite
    return transformed_df # Return transformer DataFrame




"""
this function will display a preview of the transformed data.

what this function should do:
    print a summary header
    print a readable preview for each post
    print a footer at the end
    handle empty DataFrames gracefully
"""
def display_output(df: pd.DataFrame) -> None:
    if df.empty:
        print("No posts to display.")
        return

    print("\n========== POSTS PREVIEW ==========\n")

    for row in df.iterrows():
        print(f"title: {row['title']}")
        print(f"timestamp: {row['hours']}") # used to be created_utc
        print(f"time_category: {row['time_category']}")
        print(f"day_posted: {row['day_posted']}")
        print(f"title_length: {row['title_length']}")
        print(f"title_words: {row['title_words']}")
        print(f"selftext: {row['selftext']}")
        print(f"selftext_length: {row['selftext_length']}")
        print(f"selftext_words: {row['selftext_words']}")
        print(f"has_media: {row['has_media']}")
        print(f"has_attachment: {row['has_attachment']}")
        print(f"has_flair: {row['has_flair']}")
        print(f"flair_text: {row['flair_text']}")
        print(f"has_question: {row['has_question']}")
        print(f"upvotes: {row['upvotes']}")
        print(f"num_keywords: {row['num_keywords']}")
        print(f"upvote_ratio: {row['upvote_ratio']}")
        print(f"engagement_ratio: {row['engagement_ratio']}")
        print("-" * 100)

    print("\n========== END ==========\n")
    pass


# Scrape 100 new posts, transform them, and load into SQLite
df = run_pipeline()

# # Start fresh (ignore saved cursor)
# df = run_pipeline(subreddit="UCDavis", resume=False)
