import pandas as pd
import numpy as np
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import GridSearchCV

from sklearn.tree import export_text
from sklearn.tree import plot_tree
import graphviz

import util

util.setup_logger()

def print_tree(model, df):
    r = export_text(model, feature_names=df.columns)
    print(r)

def visualise_tree(model, df, fn_out):
    dot_data = export_graphviz(
        model, 
        out_file=None,
        feature_names=df.columns,
        filled=True,
        rounded=True,
        )
    graph = graphviz.Source(dot_data) 
    graph.render(fn_out)

def main():
    
    out_dir = Path("outputs")
    mn = Path(__file__).stem

    # ------
    # data
    # -------

    target_col = "Calories"

    df_test = pd.read_csv("../../data/test.csv", index_col="id")
    df_train = pd.read_csv("../../data/train.csv", index_col="id")

    df = pd.concat([df_train, df_test])
    df = pd.get_dummies(df)
    y_raw = df.pop(target_col)
    y_log = y_raw.apply(lambda x: np.log(x + 1))

    ys_train = y_log[y_log.notna()]
    Xs_train = df[y_log.notna()]
    Xs_test = df[y_log.isna()]

    # -----
    # model
    # -----
    logging.info("using decision trees & 5-fold cv")
    param_grid = {
            #"max_depth": [30],
        "min_samples_leaf": [2],

    }
    for key, vals in param_grid.items():
        logging.info(f"{key} options: {vals}")

    model = DecisionTreeRegressor(random_state=240525)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(Xs_train, ys_train)

    logging.info(f"best params: {grid_search.best_params_}")

    # -----
    # learning curves
    # -----

    res = pd.DataFrame(grid_search.cv_results_)
    param_df = pd.json_normalize(res['params'])
    res = pd.concat([res.drop(['params'], axis=1), param_df], axis=1)
    util.plot_scans(res, param_grid, out_dir /  f"{mn}.scans.png")

    # --- 
    # do final prediction
    # ---

    model = grid_search.best_estimator_

    #print_tree(model, df)
    #visualise_tree(model, df, out_dir/f"{mn}.tree")

    ys_pred_log = model.predict(df)
    ys_pred_log = pd.Series(ys_pred_log, index=df.index)
    ys_pred = np.exp(ys_pred_log) - 1

    ys_pred_train = ys_pred[y_raw.notna()]
    ys_raw_train = y_raw[y_raw.notna()]

    out_fn = out_dir / f"{mn}.correlation.png"
    util.plot_correlation(reference=ys_raw_train, predicted=ys_pred_train, out_fn=out_fn)

    ys_pred_test = ys_pred[y_raw.isna()]
    ys_pred_test.to_csv(out_dir /f"{mn}.submission.csv")

    df_train["Calories_pred"] = ys_pred_train
    df_test["Calories_pred"] =  ys_pred_test

    print("train", df_train.describe())
    print("unique", df_train.nunique())
    print("test", df_test.describe())
    print("unique", df_test.nunique())



if __name__ == "__main__":
    main()
