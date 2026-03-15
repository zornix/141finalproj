import os
import sqlite3
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

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
    "num_keywords",
]

RESPONSE = "upvotes"


# load data from the db
def load_data(DB_PATH, TABLE_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


# make the features and the response — X is the feature cols, y is log1p(upvotes) because its right skewed
def make_xy(df) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = np.log1p(df[RESPONSE].copy())
    return X, y


# function to encode the categorical features
def one_hot_encode(df, categorical_cols) -> pd.DataFrame:
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=np.int64)


# model hyperparameters tuned manually
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




def evaluate_kfold(X_enc, y, k=5):

    model = build_model()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # cross_val_score handles the CV loop internally
    # scoring='r2' returns R-squared for each fold.
    r2_scores = cross_val_score(model, X_enc, y, cv=kf, scoring="r2")
    mae_scores = cross_val_score(model, X_enc, y, cv=kf,
                                  scoring="neg_mean_absolute_error")

    print(f"{k}-Fold CV Results:")
    for i, (r2, mae) in enumerate(zip(r2_scores, mae_scores)):
        print(f"  Fold {i+1}: R-squared={r2:.4f}   MAE={-mae:.4f}")

    print(f"\n  Mean R-squared  = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"  Mean MAE = {-mae_scores.mean():.4f} ± {mae_scores.std():.4f}")


# diagnostic plots
def plot_predicted_vs_actual(y_true_log, y_pred_log):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left panel: transformed log+1 scale (what we fit the model on)
    ax = axes[0]
    ax.scatter(y_true_log, y_pred_log, alpha=0.35, s=18, edgecolors="none")
    lo = min(y_true_log.min(), y_pred_log.min())
    hi = max(y_true_log.max(), y_pred_log.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal (y = x)")
    ax.set_xlabel("Actual  (log1p upvotes)")
    ax.set_ylabel("Predicted  (log1p upvotes)")
    ax.set_title("Predicted vs Actual — log1p scale")
    ax.legend()

    # right panel: transform back to original upvote scale for interpretation
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    ax = axes[1]
    ax.scatter(y_true_orig, y_pred_orig, alpha=0.35, s=18, edgecolors="none")
    lo = min(y_true_orig.min(), y_pred_orig.min())
    hi = max(y_true_orig.max(), y_pred_orig.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal (y = x)")
    ax.set_xlabel("Actual  (upvotes)")
    ax.set_ylabel("Predicted  (upvotes)")
    ax.set_title("Predicted vs Actual — original scale")
    ax.legend()

    fig.tight_layout()
    fig.savefig("visualizations/rf_pred_vs_actual.png", dpi=150)
    plt.close(fig)
    print("Saved visualizations/rf_pred_vs_actual.png")


# residuals plot
def plot_residuals(y_true_log, y_pred_log):
    residuals = y_true_log - y_pred_log

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left: residuals vs predicted
    ax = axes[0]
    ax.scatter(y_pred_log, residuals, alpha=0.35, s=18, edgecolors="none")
    ax.axhline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("Predicted  (log1p upvotes)")
    ax.set_ylabel("Residual  (actual - predicted)")
    ax.set_title("Residuals vs Predicted")

    # right: histogram of residuals
    ax = axes[1]
    ax.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    fig.tight_layout()
    fig.savefig("visualizations/rf_residuals.png", dpi=150)
    plt.close(fig)
    print("Saved visualizations/rf_residuals.png")


# bar chart of gini importance for top_n features
def plot_feature_importance(model, feature_names, top_n):
    # pull out gini importances, sort, take top_n and reverse for barh
    importances = np.asarray(model.feature_importances_, dtype=float)
    s = pd.Series(importances, index=feature_names)
    s = s.sort_values(ascending=False).head(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(s))))
    ax.barh(s.index, s.values)
    ax.set_title(f"RandomForest feature importance (top {min(top_n, len(s))})")
    ax.set_xlabel("Gini importance (MDI)")
    fig.tight_layout()

    fig.savefig("visualizations/rf_feature_importance.png", dpi=150)
    plt.close(fig)
    print("Saved visualizations/rf_feature_importance.png")


# permutation importance
def plot_permutation_importance(model, X_test_enc: pd.DataFrame, y_true, top_n=25):
    # shuffle each feature 15 times and see how much r-squared drops
    perm_result = permutation_importance(
        model, X_test_enc, y_true,
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )

    s = pd.Series(perm_result.importances_mean, index=X_test_enc.columns)
    s_std = pd.Series(perm_result.importances_std, index=X_test_enc.columns)
    top = s.sort_values(ascending=False).head(top_n)
    top = top[::-1]
    top_std = s_std[top.index]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.25 * len(top))))
    ax.barh(top.index, top.values, xerr=top_std.values, capsize=3)
    ax.set_title(f"Permutation importance (top {min(top_n, len(top))})")
    ax.set_xlabel("Mean decrease in R-squared when feature is shuffled")
    fig.tight_layout()

    fig.savefig("visualizations/rf_permutation_importance.png", dpi=150)
    plt.close(fig)
    print("Saved visualizations/rf_permutation_importance.png")



# train and evaluate the model
def train_random_forest() -> Tuple[RandomForestRegressor, List[str]]:
    df = load_data(DB_PATH, TABLE_NAME)
    if df.empty:
        print("No data found in the database table.")
        return None, [], None, None, None

    X, y = make_xy(df)

    categorical_cols = ["time_category", "day_posted", "flair_text"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # single train/test split for held-out evaluation + plots
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )

    X_train_enc = one_hot_encode(X_train, categorical_cols)
    X_test_enc = one_hot_encode(X_test, categorical_cols)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    model = build_model()
    model.fit(X_train_enc, y_train)

    y_pred = model.predict(X_test_enc)
    y_true = y_test.to_numpy(dtype=float)

    # error metrics on log1p scale and then we transform back for original-scale mae/rmse/r2
    mae_log = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log = r2_score(y_true, y_pred)


    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2_orig = r2_score(y_true_orig, y_pred_orig)


    print(f" Held-out test set  (train {len(X_train)} | test {len(X_test)})")
    print(f"  Categorical cols : {categorical_cols}")
    print(f"  Numeric cols     : {numeric_cols}")
    if hasattr(model, "oob_score_"):
        print(f"  OOB R-squared (log1p)   : {model.oob_score_:.4f}")
    print(f"\n  {'':>18s} {'log1p':>10s} {'original':>10s}")
    print(f"  {'MAE':>18s} {mae_log:>10.4f} {mae_orig:>10.2f}")
    print(f"  {'RMSE':>18s} {rmse_log:>10.4f} {rmse_orig:>10.2f}")
    print(f"  {'R-squared':>18s} {r2_log:>10.4f} {r2_orig:>10.4f}")

    return model, X_train_enc.columns.to_list(), y_true, y_pred, X_test_enc


# entry point
def main():
    result = train_random_forest()
    if result[0] is None:
        return

    model, feature_names, y_true, y_pred, X_test_enc = result
    evaluate_kfold(X_test_enc, y_true)
    plot_feature_importance(model, feature_names, top_n=25)
    plot_permutation_importance(model, X_test_enc, y_true, top_n=25)
    plot_predicted_vs_actual(y_true, y_pred)
    plot_residuals(y_true, y_pred)


if __name__ == "__main__":
    main()