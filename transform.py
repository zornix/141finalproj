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

# This function will clean the text by removing emojis and collapsing whitespace. 
# If the input is not a string or becomes empty after cleaning, it will return None.
def clean_text(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    text = EMOJI_PATTERN.sub("", text)
    text = " ".join(text.split())
    if text == "":
        return None
    return text


# detection helpers

# This function will detect if a post has an image and return 1 or 0.
def has_image(post_data: dict) -> int:
    try:
        # gets data on the preview, then on the image itself
        images = post_data.get("preview")
        images = post_data.get("images")
        # checks whether or not the post includes an image
        if images:
            return 1 # if there is an image, return 1 for true and 0 for false
        else:
            return 0
    except:
        print("Image error")
        return 0


# This function will detect if a post has a video and return 1 or 0.
def has_video(post_data: dict) --> int:
    try:
        videos = post_data.get("is_video") # gets data on the video
        # checks whether or not the post includes a video
        if videos:
            return 1
        else:
            return 0 # if there is a video, return 1 for true and 0 for false
    except:
        print("Video error")
        return 0
    
# checks whether or not the post includes a link
# if there is a link, return 1 for true and 0 for false
def has_link(post_data: dict) -> int:
    # return 1 if post conains a link, else 0. 
    # using the "is_self" field from reddit json to determine if the post is a self post (text only) or contains a link.
    if post_data.get("is_self") == True:
        return 0
    else:
        return 1

# checks whether or not the post includes a flair
# if there is a flair, return 1 for true and 0 for false
def has_flair(post_data: dict) --> int:
    try:
        flair = post_data.get("link_flair_text")
        if flair:
            return 1
        else:
            return 0
    except:
        print("Flair error")
        return

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

# separates the hours of the day into 4 categories
# converts the float variable into a string
def time_category(created_utc: float) --> str:
    hour = compute_hour_posted(created_utc)
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    elif 17 < hour < 24:
        return "night"
    else:
        return "late_night"

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
    
    post_image = has_image(post_data)
    post_video = has_video(post_data)
    post_link = has_link(post_data)
    flair = has_flair(post_data)
    title_length = compute_title_length(clean_title)
    selftext_length = compute_selftext_length(clean_selftext)
    title_words = compute_title_word_count(clean_title)
    selftext_words = compute_title_word_count(clean_selftext)
    time_category = time_category(updated_utc)
    day_posted = compute_day_of_week(updated_utc)
    question = is_question(clean_title)
    engagement_ratio = compute_engagement_ratio(num_comments, upvotes)
    return {
        "image": post_image,
        "link": post_link,
        "title_length": title_length,
        "selftext_length": selftext_length,
        "title_words": title_words,
        "selftext_words": selftext_words,   
        "hour_posted": hour_posted,
        "day_posted": day_posted,
        "question": question,
        "engagement_ratio": engagement_ratio
     }
    pass


"""
this function is the top-level orchestrator for transform stage.

it should convert a list of raw post dicts into a clean pandas DataFrame.
if the input list is empty, return an empty DataFrame.
"""
def transform(raw_posts: list[dict]) -> pd.DataFrame:
    # TODO: implement
    pass
