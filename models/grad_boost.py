import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import util

util.setup_logger()


def main():

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

    logging.info("using histogram gradient boosting & 3-fold cv")
    param_grid = {
            "n_estimators": [10],
            "learning_rate":[0.1],
            "criterion":["squared_error"],
            "max_depth": [3, None],
            "max_leaf_nodes":[None],
            "min_samples_leaf":[1]
    }
    for key, vals in param_grid.items():
        logging.info(f"{key} options: {vals}")


    model_out = Path(f"{mn}.pkl")
    if model_out.exists():
        with open(model_out, "rb") as f:
            grid_search = pickle.load(f)
    else:

        model = GradientBoostingRegressor(random_state=250525)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_search.fit(Xs_train, ys_train)


        logging.info(f"saving model to {model_out}")
        with open(model_out, "wb") as f:
            pickle.dump(grid_search, f)

    logging.info(f"best params: {grid_search.best_params_}")

    # -----
    # learning curves
    # -----

    res = pd.DataFrame(grid_search.cv_results_)
    param_df = pd.json_normalize(res['params'])
    res = pd.concat([res.drop(['params'], axis=1), param_df], axis=1)
    util.plot_scans(res, param_grid, f"{mn}.scans.png")


    # --- 
    # do final prediction
    # ---


    model = grid_search.best_estimator_

    ys_pred_log = model.predict(df)
    ys_pred_log = pd.Series(ys_pred_log, index=df.index)
    ys_pred = np.exp(ys_pred_log) - 1

    ys_pred_train = ys_pred[y_raw.notna()]
    ys_raw_train = y_raw[y_raw.notna()]

    out_fn = f"{mn}.correlation.png"
    util.plot_correlation(reference=ys_raw_train, predicted=ys_pred_train, out_fn=out_fn)

    ys_pred_test = ys_pred[y_raw.isna()]
    ys_pred_test.to_csv(f"{mn}.submission.csv")

    df_train["Calories_pred"] = ys_pred_train
    df_test["Calories_pred"] =  ys_pred_test

    print("train", df_train.describe())
    print("unique", df_train.nunique())
    print("test", df_test.describe())
    print("unique", df_test.nunique())



if __name__ == "__main__":
    main()
