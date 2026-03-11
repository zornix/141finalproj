import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
conn = sqlite3.connect("reddit_posts.db")
reddit_posts = pd.read_sql_query("SELECT * FROM posts", conn)

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
    plt.figure(figsize=(11,5))
    sns.heatmap(corr, annot=True)
    return plt.show()

# Distribution of upvotes per posts: summary statistics

