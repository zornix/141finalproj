"""
This file contains the transform stage of our pipeline.

We will clean raw reddit post data, detect useful flags, and add engineered features.

The output should be a pandas DataFrame that matches the db schema.
"""

import re

import pandas as pd
from datetime import datetime, timezone
from config import EMOJI_PATTERN


# cleaning helpers

"""
this function will clean text fields from reddit.

what this function should do:
    remove emojis using EMOJI_PATTERN from config.py
    collapse repeated whitespace into single spaces
    return None if input is not a string or becomes empty after cleaning
"""
def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = EMOJI_PATTERN.sub("", text)
    text = " ".join(text.split())
    if text == "":
        return None
    return text


# detection helpers

"""
this function will detect if a post has an image and return 1 or 0.

the reddit json usually has image data in preview -> images.
if that list exists and is not empty, return 1, else return 0.
"""
def has_image(post_data: dict) -> int:
    try:
        images = post_data.get("preview")
        images = post_data.get("images")
        if images:
            return 1
        else:
            return 0
    except:
        print("Image error")
        return 0


"""
this function will check if the post has a link and return 1 or 0.

reddit provides this in the "is_self" boolean field.
"""
def has_link(post_data: dict) -> int:
    # return 1 if post is not a self post 
    if post_data.get("is_self") == True:
        return 0
    else:
        return 1



# feature engineering helpers

"""
this function will return the character length of the cleaned title.
"""

def compute_title_length(cleaned_title: str) -> int:
    if cleaned_title is None:
        return 0
    return len(cleaned_title)


"""
this function will return the character length of cleaned selftext.

return 0 if selftext is None.
"""
def compute_selftext_length(cleaned_selftext: str | None) -> int:
    if cleaned_selftext is None:
        return 0
    return len(cleaned_selftext)


"""
this function will count how many words are in the cleaned title.
"""
def compute_title_word_count(cleaned_title: str) -> int:
    if cleaned_title is None:
        return 0
    return len(cleaned_title.split())


"""
this function will get the hour (0-23) from created_utc.

created_utc is a unix timestamp from reddit json.
"""
def compute_hour_posted(created_utc: float) -> int:
    return pd.to_datetime(created_utc, unit="s", utc=True).hour


"""
this function will return the weekday name from created_utc.

example outputs: Monday, Tuesday, Wednesday.
"""
def compute_day_of_week(created_utc: float) -> str:
    return pd.to_datetime(created_utc, unit="s", utc=True).day_name()


"""
this function will detect if a cleaned title looks like a question.

return 1 if title ends with "?", else 0.
"""
def is_question(cleaned_title: str) -> int:
    if cleaned_title is None or cleaned_title == "":
        return 0
    if cleaned_title[-1] == "?":
        return 1
    else:
        return 0
    

"""
this function will compute engagement ratio for a post.

ratio = num_comments / max(upvotes, 1)
use max(upvotes, 1) to avoid division by zero.
"""
def compute_engagement_ratio(num_comments: int, upvotes: int) -> float:
    ratio = num_comments / (max(upvotes, 1))
    return ratio



# top-level transform functions

"""
this function will transform one raw reddit post dict into a flat dict.

what this function should do:
    clean title and selftext        
    copy core reddit fields we need for the db
    compute all engineered features
    return one dictionary with all final columns
"""

def transform_post(post_data: dict) -> dict:
    #retrieve the data we scraped and cleaned and place it into a variable
    clean_title = clean_text(post_data.get("title"))
    clean_selftext = clean_text(post_data.get("text"))

    updated_utc = post_data.get("created_utc")
    num_comments = post_data.get("num_comments", 0)
    upvotes = post_data.get("upvotes", 0)

    pass


"""
this function is the top-level orchestrator for transform stage.

it should convert a list of raw post dicts into a clean pandas DataFrame.
if the input list is empty, return an empty DataFrame.
"""
def transform(raw_posts: list[dict]) -> pd.DataFrame:
    # TODO: implement
    pass
