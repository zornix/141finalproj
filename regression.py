import sqlite3
from xml.parsers.expat import model
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





#Lasso Regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso


def run_lasso():
    data = load_data()
    X = data.loc[:, ["title_words", "selftext_words", "attachment", "flair", "question", "num_keywords", "num_comments"]]
    y = np.log2(data.loc[:, 'upvotes']+1)
    model = Lasso(alpha=1.0)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = np.absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    model.fit(X, y)
    print(model.coef_)