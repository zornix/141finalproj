# 141finalproj
STA141b Final Project


Scraping Reddit
  What makes a winning Reddit post?
  Feature Analysis

As students, having a way to communicate with others and make connections is one of the foundations to having a fulfilling college experience. By scrapping the UCD reddit page and predicting what post would get the most engagement through upvotes we can utilize that if there is ever a need to spread information, like potential college or club events.

Our goal is to analyze the different factors that make up a post such as date posted, title, number of comments, hashtags, images or videos, hashtags, user information, links, keywords, etc. Using these variables, we will predict which combination best gets you the most number of upvotes in the UC Davis subreddit.


extract.py: This file contains the extract stage of our pipeline
load.py: This file contains the load stage of our pipeline. This file should create a table if it doesn't exist in the db, and then load the data.
