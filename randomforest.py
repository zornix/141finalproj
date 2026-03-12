import os
import sqlite3
from typing import Tuple, List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DB_PATH, TABLE_NAME


FEATURES = [
    "timestamp",
    "time_category",
    "day_posted",
    "title_words",
    "selftext_words",
    "attachment",
    "flair",
    "flair_text",
    "question",
    "num_comments",
    "num_keywords",

]

RESPONCE = "upvotes"



def load_data(DB_PATH, TABLE_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def make_xy(df) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = np.log1p(df[RESPONCE].copy())
    return X, y

def one_hot_encode(df, categorical_cols) -> pd.DataFrame:
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=np.int64)



# tuned manually, potentially need to be improved
def build_model() -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=420,
        max_depth=14,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    return model



def train_random_forest() -> Tuple[RandomForestRegressor, List[str]]:
    df = load_data(DB_PATH, TABLE_NAME)
    if df.empty:
        print("No data found in the database table.")
        return None, []


    X, y = make_xy(df)


    categorical_cols = ["time_category", "day_posted", "flair_text"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train_enc = one_hot_encode(X_train, categorical_cols)
    X_test_enc = one_hot_encode(X_test, categorical_cols)

    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    model = build_model()
    model.fit(X_train_enc, y_train)

    # evaluate the model
    y_pred = model.predict(X_test_enc)
    y_true = y_test.to_numpy(dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)


    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Categorical cols: {categorical_cols}")
    print(f"Numeric cols: {numeric_cols}")

    if hasattr(model, "oob_score_"):
        print(f"OOB R^2 (log1p scale): {model.oob_score_:.4f}")

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")

    # save the model artifact
    artifact = {
        "model": model,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "one_hot_columns": X_train_enc.columns.to_list(),
        "target_transform": "log1p",
    }
    joblib.dump(artifact, "rf/rf_upvotes.joblib")
    print(f"Saved trained model artifact to: rf/rf_upvotes.joblib")

    return model, X_train_enc.columns.to_list()


def plot_feature_importance(model, feature_names, top_n):
    importances = getattr(model, "feature_importances_", None)

    importances = np.asarray(importances, dtype=float)
    s = pd.Series(importances, index=feature_names)
    s = s.sort_values(ascending=False).head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(s))))
    ax.barh(s.index, s.values)

    ax.set_title(f"RandomForest feature importance (top {min(top_n, len(s))})")
    ax.set_xlabel("Gini importance (MDI)")
    fig.tight_layout()

    fig.savefig("rf/rf_feature_importance.png")
    plt.close(fig)


def main():
    model, feature_names = train_random_forest()
    if model is None or not feature_names:
        return

    plot_feature_importance(model, feature_names, top_n=25)


if __name__ == "__main__":
    main()

