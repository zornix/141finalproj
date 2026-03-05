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
        print(f"Title: {row['title']}")
        print(f"Time Stamp: {row['updated_utc']}")
        print(f"Time Category: {row['time_category']}")
        print(f"Day Posted: {row['day_']}")
        print(f"Title Length: {row['title_length']}")
        print(f"Title Number of Words: {row['title_words']}")
        print(f"Text: {row['selftext']}")
        print(f"Text Length: {row['selftext_length']}")
        print(f"Text Number of Words: {row['selftext_words']}")
        print(f"Image: {row['image']}")
        print(f"Video: {row['video']}")
        print(f"Attachment: {row['has_attachment']}")
        print(f"Flair: {row['has_flair']}")
        print(f"Flair Text: {row['flair_text']}")
        print(f"Question: {row['has_question']}")
        print(f"Upvotes: {row['upvotes']}")
        print(f"Number of Keywords: {row['title']}")
        print(f"Upvote Ratio: {row['upvote_ratio']}")
        print(f"Engagement Ratio: {row['title']}")
        print("-" * 100)

    print("\n========== END ==========\n")
    pass


# Scrape 100 new posts, transform them, and load into SQLite
df = run_pipeline()

# # Start fresh (ignore saved cursor)
# df = run_pipeline(subreddit="UCDavis", resume=False)
