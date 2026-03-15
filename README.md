#  STA 141B Final Project

Which features of a Reddit post on r/UCDavis correlate with the highest engagement (measured in number of upvotes)?

As students, having a way to communicate with others and make connections is one of the foundations to having a fulfilling college experience. By scrapping the UCD reddit page and predicting what post would get the most engagement through upvotes we can utilize that if there is ever a need to spread information, like potential college or club events.

Our goal is to analyze the different factors that make up a post such as date posted, title, number of comments, hashtags, images or videos, hashtags, user information, links, keywords, etc. Using these variables, we will predict which combination best gets you the most number of upvotes in the UC Davis subreddit.


Prerequisites to run the code:
```pip install -r requirements.txt```

**Pipeline**

pipeline.py: Top-level orchestrator that runs extract → transform → load and returns the transformed DataFrame.
config.py: Central constants (Reddit API headers/URL, DB path, table name, cursor file, emoji regex, batch size, sleep/time thresholds).
init.py: Runs the pipeline repeatedly until a target number of posts is processed; used to build up the dataset over multiple batches.

**ETL Pipeline**

We structured our code like an ETL pipeline with clear separation of concerns.

config.py: constants (Reddit API URL, DB path, table name, cursor file, emoji regex).

extract.py: Extract stage, fetches raw posts from Reddit’s public JSON API in batches, filters by post age, and saves cursor token.

transform.py: Transform stage, cleans raw post data (e.g. emoji removal), adds time/day features, media/flair flags, keyword counts, and outputs a DataFrame matching the DB schema.

load.py: Load stage — creates the SQLite table if missing and upserts the transformed DataFrame into the database.

pipeline.py: Pipeline orchestrator that runs extract -> transform -> load and returns the transformed DataFrame.

init.py: Runs the pipeline repeatedly until a target number of posts is processed (1000 in our case, because Reddit's listing only has 1000).

**Modeling & analysis**

regression.py: Loads data from the DB, prepares features (dummies for categorical variables, log-transformed responce -> upvotes), fits multiple OLS and Lasso, and makes residuals and Q-Q plot.

randomforest.py: Trains a RandomForest regressor on log(upvotes), runs 5-fold CV and held-out test set evaluation, and produces feature importance, permutation importance, and predicted-vs-actual values plots.

**Visualizations**
visualizations.py: EDA and summary plots, distributions (numeric/categorical), correlation heatmaps, and scatter/box plots (e.g. upvotes vs comments, title length, time of day).
