import pandas as pd
import numpy as np
import logging

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error

import util

util.setup_logger()


def main():

    target_col = "Calories"

    df_test = pd.read_csv("../../data/test.csv", index_col="id")
    df_train = pd.read_csv("../../data/train.csv", index_col="id")

    print("train", df_train.describe())
    print("unique", df_train.nunique())
    print("test", df_test.describe())
    print("unique", df_test.nunique())

    df = pd.concat([df_train, df_test])
    df = pd.get_dummies(df)
    y_raw = df.pop(target_col)
    y_log = y_raw.apply(lambda x: np.log(x + 1))

    ys_train = y_log[y_log.notna()]
    Xs_train = df[y_log.notna()]

    logging.info("using decision trees")
    dt = DecisionTreeRegressor()
    model = dt.fit(Xs_train, ys_train)

    ys_pred_log = model.predict(df)
    ys_pred_log = pd.Series(ys_pred_log, index=df.index)
    ys_pred = np.exp(ys_pred_log) - 1

    ys_pred_train = ys_pred[y_raw.notna()]
    ys_ref_train = y_raw[y_raw.notna()]

    util.plot_correlation(reference=ys_ref_train, predicted=ys_pred_train)

    ys_pred_test = ys_pred[y_raw.isna()]
    ys_pred_test.to_csv("submission.csv")


if __name__ == "__main__":
    main()
