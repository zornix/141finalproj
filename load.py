"""
This file contains the load stage of our pipeline

We could keep the data in a df or a csv file, but I think we should use a sqlite db.

This file should create the table if it doesn't exist in the db, and then load the data.
"""

import sqlite3
import pandas as pd

from config import DB_PATH, TABLE_NAME


# This function creates a table in the database if it does not already exists in db.
def create_table(conn: sqlite3.Connection) -> None:
    cur = conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                    id varchar(255),
                    timestamp float,
                    time_category varchar(255),
                    day_posted varchar(255),
                    title varchar(255),
                    title_length int,
                    title_words int,
                    selftext varchar(255),
                    selftext_length int,
                    selftext_words int,
                    image int,
                    video int,
                    has_attachment int,
                    has_flair int,
                    flair_text varchar(255),
                    has_question int,
                    upvotes int,
                    upvote_ratio float,
                    engagement_ratio float,
                    num_keywords int)""")

    conn.commit()


# This function adds rows from the transformed DataFrame to the posts table in the database.
def add_rows(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        print("There are no more rows")
        return

    df.to_sql(TABLE_NAME, conn, if_exists = "append", index = False)

    conn.commit()
    pass

# This function takes DataFrame and loads it into the database.
def load(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    if df.empty:
        print("Dataframe is empty.")
        return
    conn = sqlite3.connect(db_path)
    create_table(conn) # Create the table if it doesn't exist
    add_rows(conn, df) # Adds the rows from the df to the table
    cur = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
    rows = cur.fetchall()
    for row in rows:
        print(row)
    conn.close()
    pass
