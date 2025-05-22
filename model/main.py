import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error



def get_df(debug):

    nrows=None
    if debug:
        nrows=100


    fn = "../data/train.csv"
    df = pd.read_csv(fn,index_col="id", nrows=nrows)

    return df

    print(df.describe())


class DataHandler:

    def __init__(self, raw_df, target_col, cat_col_names):

        self.target_col = target_col
        self.cat_col_names = cat_col_names
        self.raw_df = raw_df

        cols_exclude = cat_col_names+[target_col]
        cont_col_names = [col for col in raw_df.columns if col not in cols_exclude]
        self.cont_col_names = cont_col_names

        self.ys_scaler = None
        self.Xs_scaler = None

        self.set_target_scaler()
        self.set_feature_scaler() 
        

    def set_feature_scaler(self):
   
          # scale continuous properties 
        df_cont = self.raw_df[self.cont_col_names]
        scaler = StandardScaler()
        scaler = scaler.fit(df_cont)
        self.Xs_scaler = scaler


    def set_target_scaler(self):

        # get target values
        ys_raw = self.raw_df[self.target_col]

        # log and scale them 
        ys_log = np.log(1+ys_raw)
        ys_log = pd.DataFrame({"ys":ys_log})

        scaler = StandardScaler()
        self.ys_scaler = scaler.fit(ys_log)
        
    def invert_targets(self, ys_log_norm):

        ys_log = self.ys_scaler.inverse_transform(ys_log_norm)
        ys = np.exp(ys_log) - 1

        return ys

    def transform_targets(self, ys):

        ys_log = np.log(1+ys)
        ys_log = pd.DataFrame({"ys":ys_log})
        ys_log_scaled = self.ys_scaler.transform(ys_log)
        return ys_log_scaled

    def transform_features(self, df):

        # one-hot encode categorical properties 
        # and don't scale
        df_cat = self.raw_df[self.cat_col_names]
        df_cat = self.cat_to_cont(df_cat)

        # scale continuous properties
        df_cont = self.raw_df[self.cont_col_names]
        df_cont_norm = self.Xs_scaler.transform(df_cont)
        df_cont_norm = pd.DataFrame(df_cont_norm, columns=df_cont.columns)

        # make a single df again
        dfx = pd.merge(df_cat, df_cont_norm, left_index=True, right_index=True, how='inner')

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

        df = pd.DataFrame(new_df)
        return df




def main(debug=False):

    df = get_df(debug)
    print(df.describe())

    target_col = "Calories"
    cat_col_names = ["Sex"]

    dh = DataHandler(df, target_col, cat_col_names)

    train_ys = dh.transform_targets(df[target_col])
    train_Xs = dh.transform_features(df)

    ols = linear_model.LinearRegression()
    model = ols.fit(train_Xs, train_ys)
    
    train_predictions = model.predict(train_Xs)
    train_predictions_inv_trans = dh.invert_targets(ys_log_norm=train_predictions)

    rmse_outter = root_mean_squared_error(df[target_col], train_predictions_inv_trans)
    rmse_inner = root_mean_squared_error(train_ys, train_predictions) 

    print(f"outter: {rmse_outter:.3f}, inner: {rmse_inner:.3f}")





if __name__ == "__main__":
    #main(debug=True)
    main()
