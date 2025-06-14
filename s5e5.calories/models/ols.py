import pandas as pd
import numpy as np
import logging

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def setup_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



class DataHandler:

    def __init__(self, raw_df, target_col, cat_col_names):

        self.target_col = target_col
        self.cat_col_names = cat_col_names
        self.raw_df = raw_df

        cols_exclude = cat_col_names + [target_col]
        cont_col_names = [col for col in raw_df.columns if col not in cols_exclude]
        self.cont_col_names = cont_col_names

        self.ys_scaler = None
        self.Xs_scaler = None


        logging.info(f"features one-hot-encoded ({self.cat_col_names}) or "
                     f"scale-shifted ({self.cont_col_names}).")

        self.set_target_scaler()
        self.set_feature_scaler()

    def set_feature_scaler(self):

        # scale continuous properties
        df_cont = self.raw_df[self.cont_col_names]
        scaler = StandardScaler()
        scaler = scaler.fit(df_cont)
        self.Xs_scaler = scaler
        logging.info(f"target scaler: mean {scaler.mean_} std {scaler.scale_}")

    def set_target_scaler(self):

        # get target values
        ys_raw = self.raw_df[self.target_col]

        # log and scale them
        ys_log = np.log(1 + ys_raw)
        ys_log = pd.DataFrame({"ys": ys_log})

        scaler = StandardScaler()
        self.ys_scaler = scaler.fit(ys_log)

        logging.info(f"feature scaler: mean {scaler.mean_} std {scaler.scale_}")

    def invert_targets(self, ys_log_norm):

        ys_log = self.ys_scaler.inverse_transform(ys_log_norm)
        ys = np.exp(ys_log) - 1

        return ys

    def transform_targets(self, ys):

        ys_log = np.log(1 + ys)
        ys_log = pd.DataFrame({"ys": ys_log})
        ys_log_scaled = self.ys_scaler.transform(ys_log)
        logging.info(f"targets transformed as scaleshift(np.log(1+ys))")

        return ys_log_scaled

    def transform_features(self, df):

        # one-hot encode categorical properties
        # and don't scale
        df_cat = df[self.cat_col_names]
        df_cat = self.cat_to_cont(df_cat)

        # scale continuous properties
        df_cont = df[self.cont_col_names]
        df_cont_norm = self.Xs_scaler.transform(df_cont)
        df_cont_norm = pd.DataFrame(df_cont_norm, columns=df_cont.columns, index=df_cont.index)

        # make a single df again
        dfx = pd.merge(
            df_cat, df_cont_norm, left_index=True, right_index=True, how="inner"
        )

        return dfx

    @staticmethod
    def cat_to_cont(df):

        columns = df.columns
        new_df = {}

        for col in columns:
            options = set(df[col])

            for option in options:
                new_col_name = f"{col}_{option}"
                new_df[new_col_name] = df[col].apply(lambda x: 1 if x == option else 0)

        df = pd.DataFrame(new_df, index=df.index)
        return df


def train(target_col, cat_col_names):

    fn = "../../data/train.csv"
    df = pd.read_csv(fn, index_col="id")

    dh = DataHandler(df, target_col, cat_col_names)

    train_ys = dh.transform_targets(df[target_col])
    train_Xs = dh.transform_features(df)

    ols = linear_model.LinearRegression()
    model = ols.fit(train_Xs, train_ys)

    model_msg = [f"{feat}: {coef:.3f}" for feat, coef in zip(model.feature_names_in_, model.coef_[0])]
    logging.info(f"model coeffs:"+" ".join(model_msg))


    train_preds = model.predict(train_Xs)
    train_preds_inv_trans = dh.invert_targets(ys_log_norm=train_preds)
    rmse_outer = root_mean_squared_error(df[target_col], train_preds_inv_trans)
    rmse_inner = root_mean_squared_error(train_ys, train_preds)

    logging.info(f"Errors:outer: {rmse_outer:.3f}, inner: {rmse_inner:.3f}")

    df["Calories_pred"] = train_preds_inv_trans.reshape(-1)
    print("train:")
    print(df.describe())

    plot_correlation(train_preds_inv_trans, df[target_col], train_preds, train_ys)

    return dh, model

def adjust_ax(ax):
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    min_lim = np.min([min_x, min_y])
    max_lim = np.max([max_x, max_y])

    ax.plot([min_lim, max_lim], [min_lim, max_lim], color="k", lw=1)

    ax.set_ylim(min_lim, max_lim)
    ax.set_xlim(min_lim, max_lim)
    plt.legend() 


def plot_correlation(train_pred_outer, targets_outer, train_pred_inner, targets_transformed):

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2)

    ax = fig.add_subplot(gs[0, 0])
    rmse = root_mean_squared_error(targets_outer, train_pred_outer)
    ax.scatter(targets_outer, train_pred_outer, label=f"RMSE {rmse:.3f}")
    ax.set_ylabel("Predictions, transformed back to real world values")
    ax.set_xlabel("Targets, not modified")
    ax.set_title("Real world targets")
    adjust_ax(ax)
    
    ax = fig.add_subplot(gs[0, 1])
    rmse = root_mean_squared_error(targets_transformed, train_pred_inner)
    ax.scatter(targets_transformed, train_pred_inner, label=f"RMSE {rmse:.3f}" )
    ax.set_ylabel("Predictions, not scaled back to real world")
    ax.set_xlabel("Targets, logged and normalized")
    ax.set_title("Model-side targets")
    adjust_ax(ax)


    plt.tight_layout()
    plt.savefig("ols.train.png", dpi=300)


    

def predict_test(target_col, cat_col_names, dh, model):

    fn = "../../data/test.csv"
    df = pd.read_csv(fn, index_col="id")

    test_Xs = dh.transform_features(df)
    test_pred_inner = model.predict(test_Xs)
    test_pred_outer = dh.invert_targets(test_pred_inner)

    df_out = pd.DataFrame({"Calories":test_pred_outer.reshape(-1)}, index=df.index)
    df_out.to_csv("Kaggle.PS.S5E5.csv")


    # make a single df again
    df = pd.merge(
        df, df_out, left_index=True, right_index=True, how="inner"
    )
    print("test:")
    print(df.describe())
 



def main():

    setup_logger()

    target_col = "Calories"
    cat_col_names = ["Sex"]
    
    dh, model = train(target_col, cat_col_names)
    predict_test(target_col, cat_col_names, dh, model) 



if __name__ == "__main__":
    main()
