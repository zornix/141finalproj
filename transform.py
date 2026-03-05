"""
This file contains the transform stage of our pipeline.

We will clean raw reddit post data, detect useful flags, and add engineered features.

The output should be a pandas DataFrame that matches the db schema.
"""

from pkgutil import get_data
import re

import pandas as pd
from datetime import datetime, timezone
from config import EMOJI_PATTERN


# CLEANING HELPERS

# This function will clean the text by removing emojis and collapsing whitespace.
def clean_text(text: str) -> str | None:
    if not isinstance(text, str): # If the input is not a string it will return None.
        return None
    text = EMOJI_PATTERN.sub("", text)
    text = " ".join(text.split())
    if text == "": # If the cleaned text is an empty string, it will return None.
        return None
    return text




# DETECTION HELPERS

# This function will detect if a post has an image and return 1 or 0.
def has_image(post_data: dict) -> int:
    try:
        if post_data.get("post_hint") == "image":
            return 1
        else:
            images = post_data.get("preview")
            if images and images.get("images"):
                return 1
            else:
                return 0
    except:
        print("has_image error")
        return 0


# This function will detect if a post has a video and return 1 or 0.
def has_video(post_data: dict) -> int:
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



#A self-post is a post that has no link to a website, image, or video, but instead only contains text.
# This function will detect if a post is a self post or not and return 1 or 0.
def has_attachment(post_data: dict) -> int:
    # return 1 if post conains a link, else 0.
    # using the "is_self" field from reddit json to determine if the post is a self post (text only) or contains a link.
    if post_data.get("is_self") == True:
        return 0
    else:
        return 1    # return 1 if post contains a link, else 0.


# This function will detect if a post has a flair and return a tuple (Binary has flair/not, flairtext)
def has_flair(post_data: dict) -> int:
    try:
        flair = post_data.get("link_flair_text") # gets the flair text of the post, if it exists
        if flair: # checks whether or not the post includes a flair
            return (1,flair)
        else:
            return (0, None)
    except:
        print("Flair error")
        return




# FEATURE ENGINEERING HELPERS

# This function will return the character length of the cleaned title.
def compute_title_length(cleaned_title: str) -> int:
    if cleaned_title is None:
        return 0
    return len(cleaned_title)

# This function will return the character length of the cleaned selftext.
def compute_selftext_length(cleaned_selftext: str | None) -> int:
    if cleaned_selftext is None:
        return 0
    return len(cleaned_selftext)

# This function will count how many words are in the cleaned title.
def compute_title_word_count(cleaned_title: str) -> int:
    if cleaned_title is None:
        return 0
    return len(cleaned_title.split()) # Splits the cleaned title into words and gets length


# This function will compute the hour (0-23) that the post was created, based on the created_utc from json.
def compute_hour_posted(created_utc: float) -> int:
    return pd.to_datetime(created_utc, unit="s", utc=True).hour

# This function will categorize the hour of the day into morning, afternoon, night, and late night.
def time_category(created_utc: float) -> str:
    hour = compute_hour_posted(created_utc)
    # Using Military Hours to categorize
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 17:
        return "afternoon"
    elif 17 < hour < 24:
        return "night"
    else:
        return "late_night"


# This function will compute the day of the week that the post was created, based on the created_utc from json.
def compute_day_of_week(created_utc: float) -> str:
    return pd.to_datetime(created_utc, unit="s", utc=True).day_name()

# This function checks if the title contains a question mark.
def is_question(cleaned_title: str) -> int:
    if cleaned_title is None or cleaned_title == "": # First checks to see if the cleaned title is None or empty string
        return 0
    if "?" in cleaned_title:
        return 1
    else:
        return 0

# This function will compute engagement ratio for a post.
def compute_engagement_ratio(num_comments: int, upvotes: int) -> float:
    ratio = num_comments / (max(upvotes, 1))
    return ratio

#Getting number of keywords
total_keywords = []
def count_keywords(cleaned_title: str) -> int:
    keywords = ["cheeto", "housing", "taps", "silo", "unitrans", "professor", "curve", "ship", "waitlist"]
    num_keywords = []
    for word in keywords:
        if word in cleaned_title.lower(): # Checks if the keyword is in the cleaned title, ignoring case
            num_keywords.append(1) # Adds 1 to the num_keywords list if present
            total_keywords.append(word) 
        else:
            num_keywords.append(0) # Adds 0 to the num_keywords list if not present
    return sum(num_keywords)



# TOP-LEVEL TRANSFORM FUNCTIONS

# This function will transform one raw reddit post dict into a flat dict.
def transform_post(post_data: dict) -> dict:
    #retrieve the data we scraped and cleaned and place it into a variable
    clean_title = clean_text(post_data.get("title"))
    clean_selftext = clean_text(post_data.get("text"))
    num_keywords = count_keywords(clean_title)

    updated_utc = post_data.get("created_utc") #need to make this a normal timestamp
    num_comments = post_data.get("num_comments", 0)

    # variables related to the title
    title_length = compute_title_length(clean_title)
    title_words = compute_title_word_count(clean_title)
    question = is_question(clean_title)

    # variables related to the body of text
    post_image = has_image(post_data)
    post_video = has_video(post_data)
    post_attch = has_attachment(post_data)
    post_flair = has_flair(post_data)

    selftext_length = compute_selftext_length(clean_selftext)
    selftext_words = compute_title_word_count(clean_selftext)

    # variables related to the time posted and engagement
    time_category = time_category(updated_utc)
    day_posted = compute_day_of_week(updated_utc)

    question = is_question(clean_title)
    engagement_ratio = compute_engagement_ratio(num_comments, upvotes)
    return {
        "id": post_data['id'],
        # time
        "timestamp": updated_utc,
        "time_category": time_category,
        "day_posted": day_posted,
        # title info
        "title": clean_title,
        "title_length": title_length,
        "title_words": title_words,
        # body text info
        "selftext": clean_selftext,
        "selftext_length": selftext_length,
        "selftext_words": selftext_words,
        "image": post_image,
        "video": post_video,
        "has_attachment": post_attch,
        "has_flair": post_flair[0],
        "flair_text": post_flair[1],
        "has_question": question,
        # engagement
        "upvotes": post_data['ups'],
        "upvote_ratio": post_data['upvote_ratio'],
        "engagement_ratio": engagement_ratio,
        "num_keywords": num_keywords
     } #return one dictionary with all final columns


# This function will transform a list of raw reddit post dicts into a clean pandas DataFrame. If the input list is empty, return an empty DataFrame.
def transform(raw_posts: list[dict]) -> pd.DataFrame:
    for posts in raw_posts:
        transform_post.append(transform_post(posts))
    if raw_posts is None:
        return 0
    pass
