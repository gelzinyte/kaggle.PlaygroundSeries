import pandas as pd
import numpy as np
import logging

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import GridSearchCV


import util

util.setup_logger()


def main():

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
    logging.info("using decision trees & 5-fold cv to get number of leafs")
    param_grid = {
            "max_leaf_nodes": [3, 10, 30, 100, 300, 1000, 3000, None]
    }
    for key, vals in param_grid.items():
        logging.info(f"{key} options: {vals}")

    model = DecisionTreeRegressor(random_state=240525)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(Xs_train, ys_train)

    logging.info(f"best params: {grid_search.best_params_}")


    # --- 
    # do final prediction
    # ---


    model = grid_search.best_estimator_

    ys_pred_log = model.predict(df)
    ys_pred_log = pd.Series(ys_pred_log, index=df.index)
    ys_pred = np.exp(ys_pred_log) - 1

    ys_pred_train = ys_pred[y_raw.notna()]

    util.plot_correlation(reference=ys_train, predicted=ys_pred_train)

    ys_pred_test = ys_pred[y_raw.isna()]
    ys_pred_test.to_csv("submission.csv")

    df_train["Calories_pred"] = ys_pred_train
    df_test["Calories_pred"] =  ys_pred_test

    print("train", df_train.describe())
    print("unique", df_train.nunique())
    print("test", df_test.describe())
    print("unique", df_test.nunique())


if __name__ == "__main__":
    main()
