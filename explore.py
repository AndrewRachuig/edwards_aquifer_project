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
    This function takes in a prepared and cleaned aquifer dataframe and splits it in preparation for EDA. The test size is 15% of the total data. 
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

def distribution_graphs(train):
	plt.figure(figsize=(15,6))
	plt.subplot(121)
	stats.probplot(train.water_level_elevation, plot = pylab)
	plt.subplot(122)
	train.water_level_elevation.plot(kind='hist')
	plt.show()

def monthly_distribution(train):
	ax = train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().plot.bar(width=.9, ec='black')
	plt.xticks(rotation=0)
	ax.set(title='Average Water Level by Month', xlabel='Month', ylabel='Water Level Elevation (ft)')
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().mean(), color='orange', label = "Mean Average of all Data")
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().max(), color='red', label = "Highest Average month")
	plt.axhline(train.water_level_elevation.groupby(train.water_level_elevation.index.strftime('%m-%b')).mean().min(), color='yellow', label = "Lowest Average month")
	plt.ylim(600,700)
	plt.legend()
	plt.show()

def change_by_month(train):
	# Change month to month
	y = train.water_level_elevation
	y.resample('M').mean().diff().plot(title='Average month-to-month change in water level elevation')
	plt.show()

def annual_seasonality(train):
	y = train.water_level_elevation
	y.groupby([(y.index.year//10)*10, y.index.strftime('%m-%b')]).mean().unstack(0).plot(title='Seasonal Plot')
	plt.legend(loc='lower left')
	plt.show()

def lag_plots(train):
	y = train.water_level_elevation

	plt.subplot(231)
	plt.scatter(y, y.shift(-1))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 1}$')
	plt.title('Lag of 1 Day')

	plt.subplot(232)
	plt.scatter(y, y.shift(-7))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 7}$')
	plt.title('Lag of 1 week')

	plt.subplot(233)
	plt.scatter(y, y.shift(-30))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 30}$')
	plt.title('Lag of 1 month')

	plt.subplot(234)
	plt.scatter(y, y.shift(-183))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 183}$')
	plt.title('Lag of 6 months')

	plt.subplot(235)
	plt.scatter(y, y.shift(-365))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 365}$')
	plt.title('Lag of 1 year')

	plt.subplot(236)
	plt.scatter(y, y.shift(-730))
	plt.xlabel('$y$')
	plt.ylabel('$y_{t + 730}$')
	plt.title('Lag of 2 Years')

def seasonal_decomposition_graphs(train):
	y = train.water_level_elevation.asfreq('W', method = 'bfill')
	result = sm.tsa.seasonal_decompose(y)
	result.plot()
	None