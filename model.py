import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error

plt.rcParams["figure.figsize"] = (14, 8)
import wrangle

def splits(aquifer):
    # Setting train size to be 60% of the total dataset
    train_size = int(round(aquifer.shape[0] * 0.6))

    # set validate size to be 25% of the total dataset
    validate_size = int(round(aquifer.shape[0] * 0.25))

    # Setting test size to be 15% of the total dataset. 
    test_size = int(round(aquifer.shape[0] * 0.15))

    # Checking to make sure the split worked
    if len(aquifer) == train_size + validate_size + test_size:
        print('The lengths of train, validate, test match the total length of the dataframe.')
    else:
        print('Something went wrong. The lengths of train, validate, test do not match the total length of the dataframe.')

    # Setting the datasets to have accurate frequency data
    train = train.asfreq('d', method='bfill')
    validate = validate.asfreq('d', method='bfill')
    test = test.asfreq('d', method='bfill')

    return train, validate, test