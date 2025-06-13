import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import root_mean_squared_error


from sklearn.tree import export_graphviz
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import graphviz




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



def plot_scans(df, params, fn_out):

    nplots = len(params.keys())
    side = 4

    fig = plt.figure(figsize=(side*nplots, side))
    gs = gridspec.GridSpec(1, nplots)

    plotted=False
    for idx, (scan_key, scan_vals) in enumerate(params.items()):
        ax = fig.add_subplot(gs[0, idx])

        group_by = [key for key in params.keys() if key!=scan_key] 
        if len(group_by)>1:
            for _, group_df in df.groupby(group_by):
                if len(group_df) > 1:
                    labels = [f"{gr} {np.unique(group_df[gr].values)[0]}" for gr in group_by]
                    label = " ".join(labels) 
                    ax.plot(group_df[scan_key], group_df["mean_test_score"], label=label)
                    plotted=True
        else:
            ax.plot(df[scan_key], df["mean_test_score"])
            plotted=True

     
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) 
            ax.legend()
        ax.grid()
        ax.set_xlabel(f"parameter: {scan_key}")
        ax.set_ylabel("mean_test_score")
        
    plt.tight_layout()
    if plotted:
        plt.savefig(fn_out, dpi=300)



def setup_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



def adjust_ax(ax):
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    min_lim = np.min([min_x, min_y])
    max_lim = np.max([max_x, max_y])

    ax.plot([min_lim, max_lim], [min_lim, max_lim], color="k", lw=1)

    ax.set_ylim(min_lim, max_lim)
    ax.set_xlim(min_lim, max_lim)
    plt.legend() 


def plot_correlation(reference, predicted, out_fn="correlation.png"):

    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    rmse = root_mean_squared_error(reference, predicted)
    print(f"rmse: {rmse}")
    ax.scatter(reference, predicted, label=f"RMSE {rmse:.3f}")
    ax.set_ylabel("Prediction")
    ax.set_xlabel("Reference")
    ax.set_title("RMSE")
    adjust_ax(ax)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning) 
        plt.tight_layout()
        plt.savefig(out_fn, dpi=300)


 

