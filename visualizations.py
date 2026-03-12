import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
conn = sqlite3.connect("reddit_posts.db")
reddit_posts = pd.read_sql_query("SELECT * FROM posts", conn)
reddit_posts2 = pd.read_sql_query("SELECT * FROM posts WHERE upvotes > 50 AND upvotes < 1000", conn)

# Summary Statistics
cat_columns = ['time_category', 'day_posted', 'media', 'attachment', 'flair', 'question']
num_columns = ['timestamp', 'title_length', 'title_words', 'selftext_length', 'selftext_words', 
               'upvotes', 'upvote_ratio', 'num_comments', 'num_keywords', 'score']

def numerical_summary(column):
    sns.displot(reddit_posts, x=column, kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

    numerical_stats = reddit_posts[column].describe()
    print(numerical_stats)
def categorical(column):
    sns.countplot(x=column, data=reddit_posts)
    plt.title(f'Distribution of {column}')
    plt.show()

    categorical_stats = reddit_posts[column].describe()
    print(categorical_stats)

for col in cat_columns:
    categorical(col)
for col in num_columns:
    numerical_summary(col)
    
# Correlation heatmat between variables and our response
def heatmap(table):
    newTable = table.select_dtypes(include='number') #get rid of strings
    corr = newTable.corr() # gets correlation of only numeric value in originall table
    plt.figure(figsize=(11,7))
    sns.heatmap(corr, annot=True)
    plt.show()
heatmap(reddit_posts)

# Distribution of upvotes per posts: summary statistics

# Scatter plot of upvotes vs. comments sorted by the day of the week
def daysort_upvotes_vs_comments():
    markers = {
        "Monday": "o",
        "Tuesday": "s",
        "Wednesday": "^",
        "Thursday": "D",
        "Friday": "P",
        "Saturday": "*",
        "Sunday": "X"
    }

    plt.figure(figsize=(10,7))

    for day, marker in markers.items():
        day_reddit_posts = reddit_posts[reddit_posts["day_posted"] == day]
        plt.scatter(
            day_reddit_posts["num_comments"],
            day_reddit_posts["upvotes"],
            marker=marker,
            alpha=0.7,
            s=70,
            label=day
        )

    plt.title("Upvotes vs Comments by Day Posted")
    plt.xlabel("Number of Comments")
    plt.ylabel("Number of Upvotes")

    plt.legend(title="Day Posted", bbox_to_anchor=(1.05,1))
    plt.grid(alpha=0.3)

    plt.show()

#Upvotes vs variables
def var_up(df):
    variables = ["title_words", "selftext_words", "num_keywords", "num_comments"]
    for var in variables:
        upvotes_log = np.log2(df["upvotes"]+1)
        plt.figure()
        sns.set_theme(style = 'white', palette = 'Set2') 
        plot = sns.scatterplot(
            data=df,
            x=var,
            y= upvotes_log,
        )
        plot.set_title(f"Upvotes vs {var}")
        plot.set_xlabel("upvotes")
        plot.set_ylabel(var)

        plt.show()
        plt.close() 