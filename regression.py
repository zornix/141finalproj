import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import DB_PATH, TABLE_NAME
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor



# This function loads on database and creates a pandas df
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df

# features are the variables from table
FEATURES = ["time_category", "day_posted", "title_words", "selftext_words", "attachment", "flair", "question", "num_keywords"]



def ols_diagnostics(model, X):
    residuals = model.resid
    fitted = model.fittedvalues

    # residual vs fitted plot + Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(fitted, residuals, alpha=0.4, edgecolors="k", s=20)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot")

    plt.tight_layout()
    plt.savefig("visualizations/ols_diagnostics.png", dpi=150)
    plt.close()
    print("\nDiagnostic plots saved to visualizations/ols_diagnostics.png")



# This function transform the data using log
def prepare_training_data(df: pd.DataFrame):

    y = np.log2(df["upvotes"]+1) #log transformation worked better
    #y = np.sqrt(df["upvotes"]) #square root transformation
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
    ols_diagnostics(model, X)

    # robust standard errors
    robust_model = model.get_robustcov_results('HC3')
    print(robust_model.summary())

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
    X = data.loc[:, ["title_words", "selftext_words", "attachment", "flair", "question", "num_keywords"]]
    y = np.log2(data.loc[:, 'upvotes']+1)
    model = Lasso(alpha=0.1)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    scores = np.absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    model.fit(X, y)
    coef_table = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
    })
    print(coef_table)
    print(f"intercept: {model.intercept_}")
run_lasso()