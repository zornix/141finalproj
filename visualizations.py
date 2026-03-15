import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
conn = sqlite3.connect("reddit_posts.db")

#Create two DataFrames, one with all posts and one with posts with upvotes between 50 and 1000 to remove outliers and dense numbers for better visualizations
reddit_posts = pd.read_sql_query("SELECT * FROM posts", conn)
reddit_posts2 = pd.read_sql_query("SELECT * FROM posts WHERE upvotes > 50 AND upvotes < 1000", conn)

# Summary Statistics
cat_columns = ['time_category', 'day_posted', 'media', 'attachment', 'flair', 'question']
num_columns = ['timestamp', 'title_length', 'title_words', 'selftext_length', 'selftext_words',
               'upvotes', 'upvote_ratio', 'num_comments', 'num_keywords', 'score']

def numerical_summary(column):
    plt.figure()
    sns.histplot(reddit_posts[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.savefig(f"visualizations/{column}_summary.png")
    plt.show()
    plt.close()
    numerical_stats = reddit_posts[column].describe()
    print(numerical_stats)
def categorical(column):
    plt.figure()
    sns.countplot(x=column, data=reddit_posts)
    plt.title(f'Distribution of {column}')
    plt.savefig(f"visualizations/{column}_summary.png")
    plt.show()
    plt.close()
    categorical_stats = reddit_posts[column].describe()
    print(categorical_stats)

for col in cat_columns:
    categorical(col)
for col in num_columns:
    numerical_summary(col)

# Correlation heatmat between variables and our response
def heatmap(table, filename):
    newTable = table.select_dtypes(include='number') #get rid of strings
    corr = newTable.corr() # gets correlation of only numeric value in originall table
    plt.figure(figsize=(11,7))
    sns.heatmap(corr, annot=True)
    plt.savefig(f"visualizations/{filename}")
    plt.show()
heatmap(reddit_posts, "Correlation_Heatmap")

heatmap(reddit_posts2, "Correlation_Heatmap_2")

#Summary statistics
plt.figure(figsize=(12, 6))
plot = sns.boxplot(
    data=reddit_posts2,
    x="time_category",
    y="upvotes",
)
plt.xticks(rotation=45)
plt.show()
plt.close()



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

    fig, ax = plt.subplots(figsize=(10,7))

    for day, marker in markers.items():
        day_plot_df = reddit_posts[
            (reddit_posts["day_posted"] == day) &
            (reddit_posts["upvotes"].between(50, 1000))
        ]

        plt.scatter(
            day_plot_df["num_comments"],
            day_plot_df["upvotes"],
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
    
    fig.savefig("visualizations/daysort_comments_vs_upvotes.png")
    plt.show()
    plt.close()
daysort_upvotes_vs_comments(reddit_posts)

# effect of title length on upvotes removing outliers and removing dense numbers
def titlelength_upvotes():
    plt.figure(figsize=(8,6))
    # plot_df["log_upvotes"] = np.log2(df["upvotes"]+1) for log
    plot_df = reddit_posts[reddit_posts["upvotes"].between(50, 1000)]

    plt.scatter(
        plot_df["title_length"],
        plot_df["upvotes"],
        marker="*",
        alpha=0.7,
        s=80
    )

    plt.title("Title Length Effect on Upvotes")
    plt.xlabel("Title Length")
    plt.ylabel("Upvotes")

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.show()
    plt.close()

#Upvotes vs variables
def var_up(df):
    variables = ["title_length","title_words", "selftext_words", "num_keywords", "num_comments"]
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
        plot.set_xlabel(var)
        plot.set_ylabel("upvotes (log scale)")
        plt.savefig(f"visualizations/{var}_vs_upvotes.png")
        plt.show()
        plt.close()
var_up(reddit_posts2)

#Upvotes vs Comments Categorized on Time of Day:
plt.figure()
sns.set_theme(style = 'white', palette = 'Set2')
plot = sns.scatterplot(
    data= reddit_posts2,
    x="upvotes",
    y="num_comments",
    hue="time_category",
    style="time_category"
)
plot.set_title("Reddit Post Performance by Time of Upload")
plot.set_xlabel("upvotes")
plot.set_ylabel("comments")
plt.savefig("visualizations/upvotes_comments_time.png")
plt.show()
plt.close()