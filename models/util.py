import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import root_mean_squared_error

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


 

