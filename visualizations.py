import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Summary Statistics
cat_columns = ['time_category', 'day_posted', 'media', 'attachment', 'flair', 'question']

def categorical(column):
    sns.countplot(x=column, data=reddit_posts)
    plt.title(f'Distribution of {column}')
    plt.show()

    categorical_stats = reddit_posts[column].describe()
    print(categorical_stats)

for col in cat_columns:
    categorical(col)
# Correlation heatmat between variables and our response

# Distribution of upvotes per posts: summary statistics

