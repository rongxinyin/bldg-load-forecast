import pandas as pd
import numpy as np
import os
import pathlib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn import linear_model

# Calculate model metrics
def calc_model_metrics(y, y_predicted, num_predictor):
    # calculate model metrics
    MAE = "{:.3f}".format(mean_absolute_error(y, y_predicted))
    nMAE = "{:.3f}".format(mean_absolute_error(y, y_predicted)/np.mean(y))
    MBE = "{:.3f}".format(np.mean(y_predicted-y))
    nMBE = "{:.3f}".format(np.sum(y_predicted-y)/(len(y)-1)/np.mean(y))
    RSME = "{:.3f}".format(sqrt(mean_squared_error(y, y_predicted)))
    nRSME = "{:.3f}".format(
        sqrt(mean_squared_error(y, y_predicted))/np.mean(y))
    R2 = "{:.3f}".format(r2_score(y, y_predicted))
    R2_adj = 1 - (1 - r2_score(y, y_predicted)) * \
        ((y.shape[0] - 1) / (y.shape[0] - num_predictor - 1))
    return [MAE, nMAE, MBE, nMBE, RSME, nRSME, R2, R2_adj]
    