"""
This file contains the load stage of our pipeline

We could keep the data in a df or a csv file, but I think we should use a sqlite db.

This file should create the table if it doesn't exis in the db, and then load the data.
"""

import sqlite3
import pandas as pd

from config import DB_PATH, TABLE_NAME


"""
this function will create the table if it doesn't exist in the db.
takes in a connection to the db and creates the table with the needed schema
"""

def create_table(conn: sqlite3.Connection) -> None:
    # TODO: implement
    cur = conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE_NAME}(
                    id varchar(255),
                    title varchar(255),
                    selftext varchar(255),
                    author varchar(255),
                    timestamp varchar(255),
                    upvotes int,
                    upvote_ratio int,
                    numcomments int,
                    spoiler int,
                    flair varchar(255),
                    has_Image int)""")

    conn.commit()






"""
this function will add every row from the transformed DataFrame to the posts table.

cast integer features to int beforehand, so the db stores them correctly
"""

def add_rows(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    # TODO: implement
    #conn.execute(f"""ALTER TABLE ADD ROW in df""")
    #conn.commit()

    pass



"""
this is the top-level orchestrator function for the load stage

check if the dataframe is empty
create table if it doesnt exist, upsert the data, print a summary

"""
def load(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    # TODO: implement
    pass
