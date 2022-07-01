import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import pylab
import statsmodels.api as sm

plt.rcParams["figure.figsize"] = (14, 8)

import wrangle

def train_test_split(aquifer):
    '''
    This function takes in a prepared and cleaned aquifer dataframe and splits it in preparation for EDA and modeling. The test size is 15% of the total data. 
    The function also displays a plot of the distribution of the split and returns train and test dataframes.

    Parameters: aquifer - the aquifer dataframe previously pulled in and cleaned

    Returns:    train - dataframe of the train set of aquifer ready for eda and modeling
                test - dataframe of the test set of aquifer ready for modeling
    '''
    train_size = .85
    n = aquifer.shape[0]
    test_start_index = round(train_size * n)

    train = aquifer[:test_start_index] # everything up (not including) to the test_start_index
    test = aquifer[test_start_index:] # everything from the test_start_index to the end

    plt.plot(train.index, train.water_level_elevation)
    plt.plot(test.index, test.water_level_elevation)
    return train, test