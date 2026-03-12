import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import DB_PATH, TABLE_NAME


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


FEATURES = ["time_category", "day_posted", "title_words", "selftext_words", "attachment", "flair", "question", "num_keywords"]


def prepare_training_data(df: pd.DataFrame):

    #y = np.log2(df["upvotes"]+1) #log transformation
    y = np.sqrt(df["upvotes"]) #square root transformation
    X = df[FEATURES].copy()

    for col in ["time_category", "day_posted"]:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = X.drop(columns=[col]).join(dummies)

    X = sm.add_constant(X, has_constant="add") #for intercept

    X = X.astype(float)

    return X, y


def run_regression():
    df = load_data()
    if df.empty:
        print("No data in database.")
        return None

    X, y = prepare_training_data(df)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


if __name__ == "__main__":
    run_regression()
